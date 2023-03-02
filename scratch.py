from BLIP.models.blip import blip_decoder

config = {
    "image_root": "/export/share/datasets/vision/coco/images/",
    "ann_root": "annotation",
    "coco_gt_root": "annotation/coco_gt",
    "pretrained": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth",
    "vit": "base",
    "vit_grad_ckpt": False,
    "vit_ckpt_layer": 0,
    "batch_size": 32,
    "init_lr": 1e-05,
    "image_size": 384,
    # "max_length": 240,
    "min_length": 5,
    "num_beams": 3,
    "prompt": "a stable diffusion image of ",
    "weight_decay": 0.05,
    "min_lr": 0,
    "max_epoch": 5,
}

model = blip_decoder(
    pretrained=config["pretrained"],
    image_size=config["image_size"],
    vit=config["vit"],
    vit_grad_ckpt=config["vit_grad_ckpt"],
    vit_ckpt_layer=config["vit_ckpt_layer"],
    prompt=config["prompt"],
)

print(model)
