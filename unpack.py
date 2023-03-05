import os
import zipfile

from config import Config 

zips = [z for z in os.listdir(Config.datadir) if ".zip" in z]
os.chdir(Config.datadir)

for z in zips:
    folder = z.replace(".zip", "")

    if not os.path.exists(folder):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(z.replace(".zip", ""))
    os.remove(z)