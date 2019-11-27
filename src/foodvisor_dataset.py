import json
import re
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage import io
from torch.utils.data import SubsetRandomSampler

from src.utils import Constants


class FoodVisorDataset(torch.utils.data.Dataset):
    """
    FoodVisorDataset is a custom Dataset. It is adapted to FoodVisor data convention.
    Indeed, to build this dataset, à img_annotations and a csv_mapping are needed.

    Arguments:
    ----------
        - json_annotations (str): path file for the img_annotations.json
        - csv_mapping (str): path file for the label_mapping.csv
        - root_dir (str): path folder where all images are located
        - regex_aliment (str): regex to build class. Example: with regex r"[Tt]omate(s)?" with build two classes,
        one containing only image with tomatoes, and one with everything else.
        - tranform (torchvision.transforms, default=None): Transform to apply
        - lang (str, default="fr"): lang corresponding to label ("fr" and "en" only)
    """

    def __init__(
        self,
        json_annotations: str,
        csv_mapping: str,
        root_dir: str,
        regex_aliment: str,
        augmentations: A=None,
        lang: str = "fr"
    ):
        with open(json_annotations) as f:
            self.img_annotations = json.load(f)
        self.csv_mapping = pd.read_csv(csv_mapping)
        self.root_dir = Path(root_dir)
        self.augmentations = augmentations
        self.__regex_aliment = regex_aliment
        if lang in Constants.LANG_LABEL:
            self.__lang = lang
        else:
            print("lang parameter should be one of the following :")
            for l in Constants.LANG_LABEL:
                print("   - {:s}".format(l))
            raise ValueError

    def __getitem__(self, index: int):
        img_id = list(self.img_annotations.keys())[index]
        img_name = self.root_dir / img_id

        if self.augmentations:
            image = self.augmentations(image=io.imread(img_name))['image']
        else:
            image = Image.fromarray(io.imread(img_name))

        if self.__is_aliment_present(img_id):
            return image, Constants.POSITIVE[0]
        else:
            return image, Constants.NEGATIVE[0]

    def __len__(self) -> int:
        return len(self.img_annotations.keys())

    def __get_label_for_id(self, label_id: str):
        return self.csv_mapping[self.csv_mapping[Constants.COL_LABEL_ID] == label_id][
            Constants.COL_LABEL_NAME + self.__lang
        ].values[0]

    def __is_aliment_present(self, image_id):
        func = lambda dict_box: self.__get_label_for_id(dict_box["id"])
        r = re.compile(self.__regex_aliment)
        if any(r.match(w) for w in map(func, self.img_annotations[image_id])):
            return True
        return False

    def make_weights_for_balanced_classes(self):
        count = [0] * 2
        img_ids = list(self.img_annotations.keys())
        for id in img_ids:
            if self.__is_aliment_present(id):
                count[1] += 1
            else:
                count[0] += 1
        weight_per_class = [0.] * 2
        N = float(sum(count))
        for i in range(2):
            weight_per_class[i] = N / float(count[i])
        weight = [0] * len(img_ids)
        for idx, val in enumerate(img_ids):
            if self.__is_aliment_present(val):
                weight[idx] = weight_per_class[1]
            else:
                weight[idx] = weight_per_class[0]

        return weight


class FoodDatasetLoader:
    """
    Custom loader adapted to FoodVisorDataset and with ability to build train and test loader.

    Arguments:
    ----------
        - food_dataset (FoodVisorDataset): dataset to build loaders
        - params (dict): dict with parameters for the loaders.
        Default param are : {
                "batch_size": 32,
                "validation_split": 0.2,
                "shuffle_dataset": True,
                "random_seed": 42,
                }
    """
    def __init__(self, food_dataset_train: FoodVisorDataset, food_dataset_test: FoodVisorDataset, weights, param_loader=None):
        self.food_dataset_train = food_dataset_train
        self.food_dataset_test = food_dataset_test
        self.weights = weights
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if param_loader is None:
            self.param_loader = Constants.DEFAULT_PARAM_LOADER
        else:
            self.param_loader = param_loader

    def build_loader(self):
        """
        Method to build train and test loader
        :return:
        """
        # Check if all parameter exits
        for p, value in Constants.DEFAULT_PARAM_LOADER.items():
            if p not in self.param_loader:
                self.param_loader[p] = value

        # Creating data indices for training, test and
        # validation splits with equal sample of each class:
        dataset_size = len(self.food_dataset_train)
        indices = list(range(dataset_size))
        split = int(np.floor(self.param_loader["validation_split"] * dataset_size))
        if self.param_loader["shuffle_dataset"]:
            np.random.seed(self.param_loader["random_seed"])
            np.random.shuffle(indices)
        train_indices, test_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        #train_sampler = SubsetRandomSampler(train_indices)
        #test_sampler = SubsetRandomSampler(test_indices)

        train_set = torch.utils.data.Subset(self.food_dataset_train, train_indices)
        test_set = torch.utils.data.Subset(self.food_dataset_test, test_indices)

        #print(train_set.indices)

        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=[self.weights[idx] for idx in train_indices],
            num_samples=len(train_indices),
            replacement=True
        )

        #test_sampler = SubsetRandomSampler(test_indices)

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.param_loader["batch_size"],
            sampler=train_sampler,
        )

        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=self.param_loader["batch_size"]
        )

        return train_loader, test_loader


class WeightedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
    """
    def __init__(self, dataset, indices, weights=None):
        super().__init__(dataset)
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices)

        # distribution of classes in the dataset
        if weights is None:
            label_to_count = {}
            for idx in self.indices:
                label = self._get_label(dataset, idx)
                if label in label_to_count:
                    label_to_count[label] += 1
                else:
                    label_to_count[label] = 1
            # weight for each sample
            weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                       for idx in self.indices]
            self.weights = torch.DoubleTensor(weights)
        else:
            self.weights = weights

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is FoodVisorDataset:
            return dataset[idx][1]
        else:
            raise NotImplementedError

    def __iter__(self):
        print(self.indices)
        print(torch.multinomial(
            self.weights, self.num_samples, replacement=True))
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


def plot_9_images(dataset):
    import matplotlib.pyplot as plt
    import random

    fig = plt.figure()

    idx = [random.randint(0, len(dataset)) for i in range(9)]

    for i, id in enumerate(idx):
        sample = dataset[id]

        ax = plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        ax.set_title("Food #{}\n{}".format(id, sample["label"]))
        ax.axis("off")
        ax.imshow(sample["image"])
    plt.show()
