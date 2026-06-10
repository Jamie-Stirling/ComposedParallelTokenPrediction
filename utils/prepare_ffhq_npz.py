# we're going to turn the FFHQ dataset into a .npz file
# the target file will be called ffhq_256_dataset.npz
# it will contain the following keys:
# - ims: a numpy array of shape (~70000, 256, 256, 3) containing the images, dtype=np.uint8
# - labels: a numpy array of shape (~70000, 4) containing binary attributes, dtype=np.uint8
#   - the 0 column: 1 if the image has the "smiling" attribute, 0 otherwise
#   - the 1 column: 1 if the image has the "glasses" attribute, 0 otherwise
#   - the 3 column: 1 if the image has the "male" attribute, 0 otherwise
#   - the 2 column will remain unused for historical reasons
# we'll additionally filter out images for which the attributes contain -1
import torch
from tqdm import tqdm
import numpy as np
from data_utils import FFHQImageAttributeFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
import matplotlib.pyplot as plt

if __name__ == "__main__":
    img_size = 256
    dataset = FFHQImageAttributeFolder(
        "/datasets/FFHQ",
        "/datasets/ffhq-features-dataset/json/",
        transform=None,
        n_components=None,
        return_all_attribs=True,
    )

    data_len = len(dataset)

    perm = torch.randperm(data_len)
    images = []
    labels = []
    original_indices = []

    for i in tqdm(range(data_len)):
        idx = perm[i]
        (x, l), attr = dataset[idx]
        if -1 in attr:
            continue

        images.append(np.array(x))
        # the attributes are in the order: smiling, glasses, gender; but their numbers are off
        attr_0 = 1 if attr[0] == 1 else 0  # 1 iff smiling
        attr_1 = 1 if attr[1] == 5 else 0  # 1 iff glasses
        attr_3 = 1 if attr[2] == 3 else 0  # 1 iff male
        attr_2 = 0

        labels.append([attr_0, attr_1, attr_2, attr_3])
        original_indices.append(idx.item())

        print(labels[-1])
        plt.imshow(x)
        plt.show()

    images = np.array(images)
    labels = np.array(labels).astype(np.uint8)
    original_indices = np.array(original_indices)

    # print shapes and types
    print(images.shape, images.dtype)
    print(labels.shape, labels.dtype)

    np.savez(
        "./ffhq_256_dataset.npz",
        ims=images,
        labels=labels,
        original_indices=original_indices,
    )
