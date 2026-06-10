from data_utils import FFHQImageAttributeFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    img_size = 256
    transform = Compose([Resize(img_size), CenterCrop(img_size), ToTensor()])

    for n_components in [1, 2, 3]:
        dataset = FFHQImageAttributeFolder(
            "/datasets/FFHQ",
            "/datasets/ffhq-features-dataset/json/",
            transform=transform,
            n_components=n_components,
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
        # sample a batch and display the images, and print the attributes
        print(len(dataset))
        # for some reason, the dataloader thinks the dataset has more elements than we'd get with len(dataset)
        # this is because the dataset is a subclass of torch.utils.data.Dataset, and the __len__ method is not implemented

        (images, _), attributes = next(iter(loader))
        print(images.shape)
        for i in range(4):
            plt.imshow(images[i].permute(1, 2, 0))
            print(attributes[i])
            plt.show()
