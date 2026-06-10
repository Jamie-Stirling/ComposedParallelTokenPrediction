import os, json

from PIL import Image

import matplotlib.pyplot as plt
import pandas as pd

ffhq_dir = "/datasets/FFHQ/faces/"
anno_dir = "/datasets/ffhq-features-dataset/json/"

all_img_fnames = os.listdir(ffhq_dir)
all_anno_fnames = os.listdir(anno_dir)

# sor tby the number of the image
all_img_fnames.sort(key=lambda x: int(x.split(".")[0]))
all_anno_fnames.sort(key=lambda x: int(x.split(".")[0]))


print("Number of images:", len(all_img_fnames))
print("Number of annotations:", len(all_anno_fnames))

data_rows = []

for img_fname, anno_fname in zip(all_img_fnames, all_anno_fnames):
    assert img_fname.split(".")[0] == anno_fname.split(".")[0]

    img = Image.open(os.path.join(ffhq_dir, img_fname))
    anno = json.load(open(os.path.join(anno_dir, anno_fname)))

    # if anno is list take first element
    if isinstance(anno, list) and len(anno) >= 1:
        anno = anno[0]
    else:
        continue

    smile = anno["faceAttributes"]["smile"] > 0.5
    gender = anno["faceAttributes"]["gender"]
    glasses = anno["faceAttributes"]["glasses"]

    data_rows.append(
        {
            "name": img_fname.split(".")[0],
            "smile": smile,
            "gender": gender,
            "glasses": glasses,
        }
    )

df = pd.DataFrame(data_rows)

# get some statistics: bar graphs of smile, gneder, glasses
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
df["smile"].value_counts().plot(kind="bar", ax=ax[0])
df["gender"].value_counts().plot(kind="bar", ax=ax[1])
df["glasses"].value_counts().plot(kind="bar", ax=ax[2])
plt.show()
