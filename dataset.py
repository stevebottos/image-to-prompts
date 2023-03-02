import os
import json

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, Lambda, Normalize
from torchvision.transforms.functional import InterpolationMode


from config import Config


class ImagePromptDataset(Dataset):
    def __init__(self, datadir, datafiles, imsize, limit=1):
        self.data = []
        self.embeddings_dir = "sentence_embeddings"

        # mean/std values from https://github.com/salesforce/LAVIS/blob/main/lavis/processors/blip_processors.py
        self.transforms = Compose(
            [
                Resize((imsize, imsize), interpolation=InterpolationMode.BICUBIC),
                Lambda(lambda x: x / 255.0),
                Normalize(
                    [0.48145466, 0.4578275, 0.40821073],
                    [0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

        for datafile in datafiles:
            imdir = os.path.join(datadir, os.path.splitext(datafile)[0])
            with open(f"{datadir}/{datafile}") as f:
                dataset_chunk = json.load(f)
                self.data.extend(
                    [(f"{imdir}/{k}", v["p"]) for k, v in dataset_chunk.items()]
                )
        if limit:
            self.data = self.data[:limit]

    def __len__(self):
        return len(self.data)

    def _embeddings_exist(self):
        if os.path.exists(self.embeddings_dir):
            return True

    # NOTE: Don't need this right now but will later
    # def _prompts_to_embeddings(self):
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    #     st_model = SentenceTransformer("all-MiniLM-L6-v2").to(device)

    #     if not os.path.exists(self.embeddings_dir):
    #         os.mkdir(self.embeddings_dir)

    #     for d in tqdm(self.data):
    #         embeddingfile = self._impath_to_embedpath(d[0])
    #         prompt = d[1]
    #         embedding = st_model.encode(prompt).flatten()
    #         np.save(embeddingfile, embedding)

    def __getitem__(self, idx):
        prompt = self.data[idx][1]
        image = self.transforms(read_image(self.data[idx][0]).float())
        return image, prompt


def get_datasets(train_files, val_files, train_limit=None, val_limit=None):
    train_dataset = ImagePromptDataset(
        datadir=Config.datadir,
        datafiles=train_files,
        imsize=Config.image_size,
        limit=train_limit,
    )

    val_dataset = ImagePromptDataset(
        datadir=Config.datadir,
        datafiles=val_files,
        imsize=Config.image_size,
        limit=val_limit,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
    )

    return train_dataloader, val_dataloader


if __name__ == "__main__":
    from config import Config

    dset = ImagePromptDataset(datadir=Config.datadir)

    image, embed = dset[0]
    print(image.dtype, embed.dtype)
