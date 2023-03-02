class Config:
    # data stuff
    datadir = "/mnt/e/datasets/stable_diffusion_image_prompts"
    train_datafiles = [
        "part-000001.json",
        "part-000002.json",
        "part-000003.json",
        "part-000004.json",
        "part-000005.json",
        "part-000006.json",
        "part-000007.json",
        "part-000008.json",
        "part-000009.json",
    ]
    val_datafiles = ["part-000010.json"]
    test_datafiles = ["test.json"]

    # model stuff
    image_size = 384
    pretrained = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth"
    max_tokenizer_length = 256
