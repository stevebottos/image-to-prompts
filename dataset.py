import os
import glob
import json
import time

from sentence_transformers import SentenceTransformer
import torch

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, Lambda, Normalize
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
import numpy as np


class ImagePromptDataset(Dataset):
    def __init__(self, datadir, limit=1, imagesize=384):
        data = glob.glob(f"{datadir}/*.json")
        self.data = []
        self.embeddings_dir = "sentence_embeddings"

        # mean/std values from https://github.com/salesforce/LAVIS/blob/main/lavis/processors/blip_processors.py
        self.transforms = Compose(
            [
                Resize((imagesize, imagesize), interpolation=InterpolationMode.BICUBIC),
                Lambda(lambda x: x / 255.0),
                Normalize(
                    [0.48145466, 0.4578275, 0.40821073],
                    [0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

        for i, d in enumerate(data):
            imdir = os.path.splitext(d)[0]
            with open(d) as f:
                dataset_chunk = json.load(f)
            self.data.extend(
                [(f"{imdir}/{k}", v["p"]) for k, v in dataset_chunk.items()]
            )

            if (i + 1) == limit:
                break

        if not self._embeddings_exist():
            print("Pre-computing prompt embeddings...")
            self._prompts_to_embeddings()

    def __len__(self):
        return len(self.data)

    def _embeddings_exist(self):
        if os.path.exists(self.embeddings_dir) and (
            len(os.listdir(self.embeddings_dir)) == len(self.data)
        ):
            return True

    def _prompts_to_embeddings(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st_model = SentenceTransformer("all-MiniLM-L6-v2").to(device)

        if not os.path.exists(self.embeddings_dir):
            os.mkdir(self.embeddings_dir)

        for d in tqdm(self.data):
            embeddingfile = self._impath_to_embedpath(d[0])
            prompt = d[1]
            embedding = st_model.encode(prompt).flatten()
            np.save(embeddingfile, embedding)

    def _impath_to_embedpath(self, impath):
        return os.path.join(
            self.embeddings_dir, os.path.splitext(os.path.basename(impath))[0] + ".npy"
        )

    def __getitem__(self, idx):
        impath = self.data[idx][0]
        prompt_raw = self.data[idx][1]
        embeddingfile = self._impath_to_embedpath(impath)
        image = self.transforms(read_image(self.data[idx][0]).float())
        prompt_embedding = torch.tensor(np.load(embeddingfile)).float()
        return image, prompt_embedding, prompt_raw


if __name__ == "__main__":
    from config import Config

    dset = ImagePromptDataset(datadir=Config.datadir)

    image, embed = dset[0]
    print(image.dtype, embed.dtype)
