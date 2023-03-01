import os

import torch
from torch.utils.data import DataLoader
from lavis.models import load_model_and_preprocess

from dataset import ImagePromptDataset
from config import Config


device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, _ = load_model_and_preprocess(
    name="blip_caption", model_type="base_coco", is_eval=False, device=device
)
# print(dir(model))
# exit()

train_dataset = ImagePromptDataset(
    datadir=Config.datadir,
)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)


for image, prompt_embed, prompt_raw in train_dataloader:
    image = image.to(device)
    image_embeddings = model.visual_encoder(image)
    text_embeddings = model.text_decoder(image_embeddings)
    print(text_embeddings.shape)

    exit()
    # print(model.generate({"image": image}))
    # print(prompt_raw)
    # exit()
