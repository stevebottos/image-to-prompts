import os 
import glob 
class Config:
    # data stuff
    datadir = "data"
    all_datafiles = [os.path.basename(df) for df in glob.glob(f"{datadir}/*.json") if "test" not in df]
    train_datafiles = all_datafiles[:-1]
    val_datafiles = [all_datafiles[-1]]
    test_datafiles = ["test.json"]

    # model stuff
    image_size = 384
    pretrained = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth"
    max_tokenizer_length = 256
