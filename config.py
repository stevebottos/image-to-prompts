class Config:
    # data stuff
    datadir = "/mnt/e/datasets/stable_diffusion_image_prompts"
    train_datafiles = ["part-000001.json", "part-000002.json", "part-000003.json"]
    val_datafiles = ["part-000004.json"]

    # model stuff
    image_size = 384
    pretrained = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth"
    max_tokenizer_length = 256
