import numpy as np
import torch
from tqdm import tqdm

from dataset import get_datasets

from BLIP_models.blip import blip_decoder
from config import Config

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataloader, val_dataloader = get_datasets(
        train_files=Config.train_datafiles,
        val_files=Config.val_datafiles,
        val_limit=250,
    )

    model = blip_decoder(
        pretrained=Config.pretrained,
        image_size=Config.image_size,
        vit="base",
        vit_grad_ckpt=False,
        vit_ckpt_layer=0,
        prompt="",
        max_tokenizer_length=Config.max_tokenizer_length,
    ).to(device)

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=1e-05,
        weight_decay=0.05,
    )

    for epoch in range(50):
        losses = []
        model.train()
        for image, prompt_raw in tqdm(train_dataloader):
            image = image.to(device)
            loss = model(image, prompt_raw)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        t_loss = np.nanmean(losses)

        losses = []
        model.eval()
        with torch.no_grad(), open(f"check/epoch_{epoch}.txt", "w") as f:
            for image, prompt_raw in tqdm(val_dataloader):
                # human readable prompt
                pred_prompt = model.generate(
                    image,
                    sample=True,
                    min_length=1,
                    max_length=Config.max_tokenizer_length,
                )
                f.write(f"TRUE: {prompt_raw[0]}\nPRED: {pred_prompt[0]}\n\n")

                loss = model(image, prompt_raw)
                losses.append(loss.item())
        v_loss = np.nanmean(losses)
        print(f"T: {round(t_loss, 4)}\nV: {round(t_loss, 4)}")
