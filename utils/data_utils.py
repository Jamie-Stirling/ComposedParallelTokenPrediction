import os
import json
import math
import imageio
import yaml
import torch
import torchvision
from torch.utils.data.dataset import Subset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
    Lambda,
    RandomResizedCrop,
)
from PIL import Image
import random
from torch.utils.data import Dataset
import numpy as np


class BigDataset(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.image_paths = os.listdir(folder)

    def __getitem__(self, index):
        path = self.image_paths[index]
        img = imageio.imread(self.folder + path)
        img = torch.from_numpy(img).permute(2, 0, 1)  # -> channels first
        return img

    def __len__(self):
        return len(self.image_paths)


class NoClassDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, length=None):
        self.dataset = dataset
        self.length = length if length is not None else len(dataset)

    def __getitem__(self, index):
        img = self.dataset[index][0]
        if isinstance(img, tuple):
            img = img[0]
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)
        return img.mul(255).clamp_(0, 255).to(torch.uint8)

    def __len__(self):
        return self.length


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def get_default_dataset_paths():
    with open("datasets.yml") as yaml_file:
        read_data = yaml.load(yaml_file, Loader=yaml.FullLoader)

    paths = {}
    for i in range(len(read_data)):
        paths[read_data[i]["dataset"]] = read_data[i]["path"]

    return paths


def train_val_split(dataset, train_val_ratio):
    indices = list(range(len(dataset)))
    split_index = int(len(dataset) * train_val_ratio)
    train_indices, val_indices = indices[:split_index], indices[split_index:]
    train_dataset, val_dataset = Subset(dataset, train_indices), Subset(
        dataset, val_indices
    )
    return train_dataset, val_dataset


class FFHQImageAttributeFolder(torchvision.datasets.ImageFolder):
    def __init__(
        self,
        root,
        annotations_root,
        transform=None,
        target_transform=None,
        loader=None,
        is_valid_file=None,
        n_components=None,
        return_all_attribs=False,
    ):
        super().__init__(root, transform, target_transform, is_valid_file=is_valid_file)
        self.annotations_root = annotations_root
        self.n_components = n_components
        self.partition_indices = None
        self.partition_attrib_choices = None
        self.return_all_attribs = return_all_attribs
        if n_components is not None:
            self.partition_indices = []
            self.partition_attrib_choices = []
            with open(f"./ffhq_{n_components}_partition.txt") as f:
                for line in f:
                    idx, *choices = line.split()
                    idx = int(idx)
                    choices = [int(c) for c in choices]
                    self.partition_indices.append(idx)
                    self.partition_attrib_choices.append(choices)
            self.partition_indices = torch.tensor(self.partition_indices)
            self.partition_attrib_choices = torch.tensor(self.partition_attrib_choices)

    def __getitem__(self, index):
        img_index = index
        if self.partition_indices is not None:
            img_index = self.partition_indices[index]

        img = super().__getitem__(img_index)
        img_path = self.imgs[img_index][0]
        img_name = os.path.split(img_path)[-1].split(".")[0]
        annotation_path = os.path.join(self.annotations_root, img_name + ".json")
        anno = json.load(open(annotation_path))
        # convert annotation json to tensor with 3 indices

        # if anno is list take first element
        if isinstance(anno, list) and len(anno) >= 1:
            anno = anno[0]
        else:
            if self.n_components is None and not self.return_all_attribs:
                return img, torch.tensor([-1])
            else:
                return img, torch.tensor([-1, -1, -1])[: self.n_components]

        smile = anno["faceAttributes"]["smile"] > 0.5
        gender = anno["faceAttributes"]["gender"]
        glasses = anno["faceAttributes"]["glasses"]

        smile_int = 1 if smile else 0
        gender_int = {"female": 2, "male": 3}[gender]
        glasses_int = {
            "NoGlasses": 4,
            "ReadingGlasses": 5,
            "Sunglasses": 5,
            "SwimmingGoggles": 5,
        }[glasses]

        if self.n_components is None and not self.return_all_attribs:
            # we're training, so choose a random number from 0 to 2 inclusive
            chosen_attrib_idx = random.randint(0, 2)
            attrib = torch.tensor(
                [[smile_int, gender_int, glasses_int][chosen_attrib_idx]]
            )
        elif self.return_all_attribs:
            attrib = torch.tensor([smile_int, glasses_int, gender_int])
        else:
            # we're evaluating, so choose the subset indexed by the partition choice
            all_attribs = torch.tensor([smile_int, gender_int, glasses_int])
            attrib = all_attribs[self.partition_attrib_choices[index]]
        return img, attrib

    def __len__(self) -> int:
        if self.partition_indices is not None:
            return len(self.partition_indices)
        return super().__len__()


class CLEVRImageAttributeFolder(torchvision.datasets.ImageFolder):
    def __init__(
        self,
        root,
        annotations_path,
        transform=None,
        target_transform=None,
        loader=None,
        is_valid_file=None,
    ):
        super().__init__(root, transform, target_transform, is_valid_file=is_valid_file)
        self.annotations_path = annotations_path
        # load annotations from file
        with open(annotations_path) as f:
            self.annotations = json.load(f)["scenes"]

        # index by image filename
        self.annotations = {a["image_filename"]: a for a in self.annotations}

    def __getitem__(self, index, return_annotations=False):
        img = super().__getitem__(index)
        img_path = self.imgs[index][0]
        img_fname = os.path.split(img_path)[-1]

        color_offset = 1024
        color_range = 8
        material_offset = color_offset + color_range
        material_range = 2
        shape_offset = material_offset + material_range
        shape_range = 3
        size_offset = shape_offset + shape_range
        size_range = 2
        # TODO: move these into constructor
        colors = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
        materials = ["rubber", "metal"]
        shapes = ["sphere", "cube", "cylinder"]
        sizes = ["large", "small"]

        # get annotation for image
        annotation = None
        if img_fname in self.annotations:
            annotation = self.annotations[img_fname]
            # select a random object from the image
            num_objects = len(annotation["objects"])
            obj_idx = torch.randint(num_objects, (1,))[0]
            obj = annotation["objects"][obj_idx]
            color_int = colors.index(obj["color"]) + color_offset
            material_int = materials.index(obj["material"]) + material_offset
            shape_int = shapes.index(obj["shape"]) + shape_offset
            size_int = sizes.index(obj["size"]) + size_offset

            if return_annotations:
                return (
                    img,
                    torch.tensor([color_int, material_int, shape_int, size_int]),
                ), annotation
        else:
            print(f"Warning: no annotation for image {img_fname}")
            color_int = -1
            material_int = -1
            shape_int = -1
            size_int = -1

        return (img, torch.tensor([color_int, material_int, shape_int, size_int]))


class CLEVRImageObjectPositionFolder(torchvision.datasets.ImageFolder):
    def __init__(
        self,
        root,
        annotations_path,
        transform=None,
        target_transform=None,
        loader=None,
        is_valid_file=None,
    ):
        super().__init__(root, transform, target_transform, is_valid_file=is_valid_file)
        self.annotations_path = annotations_path
        # load annotations from file
        with open(annotations_path) as f:
            self.annotations = json.load(f)["scenes"]

        # index by image filename
        self.annotations = {a["image_filename"]: a for a in self.annotations}

    def __getitem__(self, index, return_annotations=False):
        img = super().__getitem__(index)[0]
        img_path = self.imgs[index][0]
        img_fname = os.path.split(img_path)[-1]

        # get annotation for image
        annotation = None
        if img_fname in self.annotations:
            annotation = self.annotations[img_fname]
            # select a random object from the image
            num_objects = len(annotation["objects"])
            obj_idx = torch.randint(num_objects, (1,))[0]
            obj = annotation["objects"][obj_idx]
            position = torch.tensor(obj["pixel_coords"])

            if return_annotations:
                return (img, position), annotation
        else:
            print(f"Warning: no annotation for image {img_fname}")
            position = torch.tensor([-1, -1])
            num_objects = -1

        # pad top and bottom of image to make it square
        # first, turn PIL image into tensor
        # IMG IS NOT YET A TENSOR
        # get image size

        # normalize position
        position = position.float() / 480

        return (img, position[:2], num_objects)


def clevr_pad_fn(img):
    img_size = img.shape[-1]
    # calculate padding
    pad = (img_size - img.shape[-2]) // 2
    # scale down y position accordingly
    # position[1] += pad
    # normalize position to [0, 1]
    # position = position.float() / img_size
    # pad image
    img_shape = img.shape
    img = torch.nn.functional.pad(
        img.reshape(1, *img_shape), (0, 0, pad, pad), mode="replicate"
    )
    img = img.reshape(3, img_size, img_size)
    return img


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


class Clevr2DPosDataset(Dataset):
    def __init__(
        self,
        resolution,
        data_path,
        use_captions=False,
        random_crop=False,
        random_flip=False,
    ):
        self.resolution = resolution
        self.use_captions = use_captions
        self.random_crop = random_crop
        self.random_flip = random_flip

        print("Using random crop:", random_crop)
        print("Using random flip:", random_flip)

        data = np.load(data_path)
        print(f"loading data from {data_path}...")
        print(f'using {"captions" if use_captions else "numeric labels"}...')
        self.ims, self.labels = data["ims"], data["coords_labels"]

    def __len__(self):
        return self.ims.shape[0]

    def __getitem__(self, index):
        image = Image.fromarray(self.ims[index]).convert("RGB")
        label = self.labels[index]

        if self.random_crop:
            arr = random_crop_arr(image, self.resolution)
        else:
            arr = center_crop_arr(image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 255.0
        if self.use_captions:
            out_dict = {"caption": f"[{label[0]:.2f}, {label[1]:.2f}]"}
        else:
            out_dict = {"y": label}
            masks = random.random() > 0.1
            out_dict.update(dict(masks=masks))

        return np.transpose(arr, [2, 0, 1]), out_dict


class ClevrDataset(Dataset):
    def __init__(
        self,
        resolution,
        data_path,
        use_captions=False,
        random_crop=False,
        random_flip=False,
    ):
        self.resolution = resolution
        self.use_captions = use_captions
        self.random_crop = random_crop
        self.random_flip = random_flip

        data = np.load(data_path)
        print(f"loading data from {data_path}...")
        print(f'using {"captions" if use_captions else "numeric labels"}...')
        self.ims, self.labels = data["ims"], data["labels"]

        # caption mapping
        colors_to_idx = {
            "gray": 0,
            "red": 1,
            "blue": 2,
            "green": 3,
            "brown": 4,
            "purple": 5,
            "cyan": 6,
            "yellow": 7,
            "none": 8,
        }
        shapes_to_idx = {"cube": 0, "sphere": 1, "cylinder": 2, "none": 3}
        materials_to_idx = {"rubber": 0, "metal": 1, "none": 2}
        sizes_to_idx = {"small": 0, "large": 1, "none": 2}
        relations_to_idx = {
            "left": 0,
            "right": 1,
            "front": 2,
            "behind": 3,
            "below": 4,
            "above": 5,
            "none": 6,
        }

        self.label_description = {
            "left": "to the left of",
            "right": "to the right of",
            "behind": "behind",
            "front": "in front of",
            "above": "above",
            "below": "below",
        }

        self.colors = list(colors_to_idx.keys())
        self.shapes = list(shapes_to_idx.keys())
        self.materials = list(materials_to_idx.keys())
        self.sizes = list(sizes_to_idx.keys())
        self.relations = list(relations_to_idx.keys())

    def __len__(self):
        return self.ims.shape[0]

    def __getitem__(self, index):
        image = Image.fromarray(self.ims[index]).convert("RGB")
        label = self.labels[index]

        if self.random_crop:
            arr = random_crop_arr(image, self.resolution)
        else:
            arr = center_crop_arr(image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 255.0
        if self.use_captions:
            out_dict = {"caption": self.get_caption(label)}
        else:
            out_dict = {"y": label}
            masks = random.random() > 0.05
            out_dict.update(dict(masks=masks))

        return np.transpose(arr, [2, 0, 1]), out_dict

    def get_caption(self, label):
        text_label = []
        for i in range(2):  # take the two objects (two of the objects...)
            shape, size, color, material, pos = label[i * 5 : i * 5 + 5]
            obj = " ".join(
                [
                    self.sizes[size],
                    self.colors[color],
                    self.materials[material],
                    self.shapes[shape],
                ]
            ).strip()
            text_label.append(obj)
        relation = self.relations[label[-1]]  # there's only one relation per image
        if "none" in relation:
            return text_label[0]
        else:
            return f"{text_label[0]} {self.label_description[relation]} {text_label[1]}"


def get_datasets(
    dataset_name,
    img_size,
    get_val_dataset=False,
    get_flipped=False,
    train_val_split_ratio=0.95,
    custom_dataset_path=None,
    extra_augs=True,
    n_components=None,
    do_resized_crop=True,
):

    transform = Compose([Resize(img_size), CenterCrop(img_size), ToTensor()])
    transform_with_flip = Compose(
        [
            Resize(img_size),
            CenterCrop(img_size),
            RandomHorizontalFlip(p=1.0),
            ToTensor(),
        ]
    )

    if do_resized_crop:
        transform = Compose(
            [
                Resize(img_size),
                CenterCrop(img_size),
                RandomResizedCrop(img_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
                ToTensor(),
            ]
        )
        transform_with_flip = Compose(
            [
                Resize(img_size),
                CenterCrop(img_size),
                RandomHorizontalFlip(p=1.0),
                RandomResizedCrop(img_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
                ToTensor(),
            ]
        )
    """if dataset_name == "clevr":
        transform = Compose([ToTensor(), Lambda(clevr_pad_fn), Resize(img_size)])"""
    default_paths = get_default_dataset_paths()

    if dataset_name in default_paths:
        dataset_path = default_paths[dataset_name]
    elif dataset_name == "custom":
        if custom_dataset_path:
            dataset_path = custom_dataset_path
        else:
            raise ValueError("Custom dataset selected, but no path provided")
    else:
        raise ValueError(
            f"Invalid dataset chosen: {dataset_name}. To use a custom dataset, set --dataset \
            flag to 'custom'."
        )

    if dataset_name == "churches":
        train_dataset = torchvision.datasets.LSUN(
            dataset_path, classes=["church_outdoor_train"], transform=transform
        )
        if get_flipped:
            train_dataset_flip = torchvision.datasets.LSUN(
                dataset_path,
                classes=["church_outdoor_train"],
                transform=transform_with_flip,
            )
        if get_val_dataset:
            val_dataset = torchvision.datasets.LSUN(
                dataset_path, classes=["church_outdoor_val"], transform=transform
            )

    elif dataset_name == "bedrooms":
        train_dataset = torchvision.datasets.LSUN(
            dataset_path,
            classes=["bedroom_train"],
            transform=transform,
        )
        if get_val_dataset:
            val_dataset = torchvision.datasets.LSUN(
                dataset_path,
                classes=["bedroom_val"],
                transform=transform,
            )

        if get_flipped:
            train_dataset_flip = torchvision.datasets.LSUN(
                dataset_path,
                classes=["bedroom_train"],
                transform=transform_with_flip,
            )

    elif dataset_name == "ffhq":
        """train_dataset = torchvision.datasets.ImageFolder(
            dataset_path,
            transform=transform,
        )"""
        train_dataset = FFHQImageAttributeFolder(
            dataset_path,
            "/datasets/ffhq-features-dataset/json/",
            transform=transform if not get_flipped else transform_with_flip,
            n_components=n_components,
        )

        if get_val_dataset:
            train_dataset, val_dataset = train_val_split(
                train_dataset, train_val_split_ratio
            )
            if get_flipped:
                train_dataset_flip, _ = train_val_split(
                    train_dataset_flip, train_val_split_ratio
                )
    elif dataset_name == "clevr_comp":
        train_dataset = CLEVRImageAttributeFolder(
            dataset_path,
            "/datasets/CLEVR_CoGenT_v1.0/scenes/CLEVR_trainA_scenes.json",
            transform=transform if not get_flipped else transform_with_flip,
        )
        if get_val_dataset:
            train_dataset, val_dataset = train_val_split(
                train_dataset, train_val_split_ratio
            )
    elif dataset_name == "clevr":
        train_dataset = CLEVRImageObjectPositionFolder(
            "/datasets/CLEVR_v1.0/images/train",
            "/datasets/CLEVR_v1.0/scenes/CLEVR_train_scenes.json",
            transform=transform if not get_flipped else transform_with_flip,
        )
        if get_val_dataset:
            val_dataset = CLEVRImageObjectPositionFolder(
                "/datasets/CLEVR_v1.0/images/val",
                "/datasets/CLEVR_v1.0/scenes/CLEVR_val_scenes.json",
                transform=transform,
            )
    elif dataset_name == "clevr_pos":
        if n_components is None:
            train_dataset = Clevr2DPosDataset(
                resolution=img_size,
                data_path="/datasets/clevr_pos_data_128_30000.npz",
                use_captions=False,
                random_crop=extra_augs,
                random_flip=extra_augs,
            )
        else:
            train_dataset = Clevr2DPosDataset(
                resolution=img_size,
                data_path=f"/datasets/clevr_pos_5000_{n_components}.npz",
                use_captions=False,
                random_crop=extra_augs,
                random_flip=extra_augs,
            )
    elif dataset_name == "clevr_rel":
        if n_components is None:
            train_dataset = ClevrDataset(
                resolution=img_size,
                data_path="/datasets/clevr_training_data_128.npz",
                use_captions=False,
                random_crop=extra_augs,
                random_flip=extra_augs,
            )
        else:
            train_dataset = ClevrDataset(
                resolution=img_size,
                data_path=f"/datasets/clevr_generation_{n_components}_relations.npz",
                use_captions=False,
                random_crop=extra_augs,
                random_flip=extra_augs,
            )

    if get_flipped:
        train_dataset = torch.utils.data.ConcatDataset(
            [train_dataset, train_dataset_flip]
        )

    if not get_val_dataset:
        val_dataset = None

    return train_dataset, val_dataset


def get_data_loaders(
    dataset_name,
    img_size,
    batch_size,
    get_flipped=False,
    train_val_split_ratio=0.95,
    custom_dataset_path=None,
    num_workers=4,
    drop_last=False,
    shuffle=True,
    extra_augs=True,
    get_val_dataloader=False,
):

    train_dataset, val_dataset = get_datasets(
        dataset_name,
        img_size,
        get_flipped=get_flipped,
        get_val_dataset=get_val_dataloader,
        train_val_split_ratio=train_val_split_ratio,
        custom_dataset_path=custom_dataset_path,
        extra_augs=extra_augs,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=num_workers,
        sampler=None,
        shuffle=shuffle,
        batch_size=batch_size,
        drop_last=drop_last,
    )
    if get_val_dataloader:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            num_workers=num_workers,
            sampler=None,
            shuffle=shuffle,
            batch_size=batch_size,
            drop_last=drop_last,
        )
    else:
        val_loader = None

    return train_loader, val_loader


def test_clevr_img_position():
    import matplotlib.pyplot as plt

    dataset = Clevr2DPosDataset(
        resolution=128,
        data_path="/datasets/clevr_pos_data_128_30000.npz",
        use_captions=True,
        random_crop=True,
        random_flip=True,
    )
    # get some random training images
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    for i, (images, labels) in enumerate(data_loader):
        print(images.shape)
        print(labels)
        plt.imshow(images[0].permute(1, 2, 0))
        plt.show()


def test_clevr_rel():
    dataset = ClevrDataset(
        resolution=128,
        data_path="/datasets/clevr_training_data_128.npz",
        use_captions=False,
        random_crop=True,
        random_flip=True,
    )
    import matplotlib.pyplot as plt

    # get some random training images
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    for i, (images, labels) in enumerate(data_loader):
        print(images.shape)
        print(labels)
        plt.imshow(images[0].permute(1, 2, 0))
        plt.show()


if __name__ == "__main__":
    test_clevr_rel()
