from data_utils import FFHQImageAttributeFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    img_size = 256
    transform = Compose([Resize(img_size), CenterCrop(img_size), ToTensor()])
    dataset = FFHQImageAttributeFolder(
        "/datasets/FFHQ",
        "/datasets/ffhq-features-dataset/json/",
        transform=transform,
        n_components=None,
    )

    # filter out images for which the attribute contains one or more -1
    idx_lists = [[], [], []]
    choice_lists = [[], [], []]
    target_per_list = 5000
    counter = 0
    # perm of the dataset indices
    data_len = len(dataset)

    perm = torch.randperm(data_len)

    for i in range(len(idx_lists)):
        while len(idx_lists[i]) < target_per_list:
            idx = perm[counter]
            counter += 1
            (x, l), attr = dataset[idx]
            if -1 in attr:  # missing attribute: skip
                continue

            n_components = i + 1
            # choose n_components indices at random from [0, 1, 2]

            chosen_indices = torch.randperm(3)[:n_components].tolist()

            idx_lists[i].append(idx)
            choice_lists[i].append(chosen_indices)
            print(chosen_indices)

    for i in range(len(idx_lists)):
        with open(f"./ffhq_{i+1}_partition.txt", "w") as f:
            for j, idx in enumerate(idx_lists[i]):
                f.write(f"{idx} ")
                f.write(" ".join([str(x) for x in choice_lists[i][j]]))
                f.write("\n")
    print("done")
