import json
import random
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
        - json_annotations (dict): dictionnary of the img_annotations.json
        - csv_mapping (str): path file for the label_mapping.csv
        - root_dir (str): path folder where all images are located
        - regex_aliment (str): regex to build class. Example: with regex r"[Tt]omate(s)?" with build two classes,
        one containing only image with tomatoes, and one with everything else.
        - augmentations (albumentation, default=None): Transform to apply using albumentation
        - lang (str, default="fr"): lang corresponding to label ("fr" and "en" only)
    """

    def __init__(
        self,
        json_annotations: dict,
        csv_mapping: str,
        root_dir: str,
        regex_aliment: str,
        augmentations: A = None,
        lang: str = "fr",
    ):
        self.img_annotations = json_annotations
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

        # For faster computation, let's build a dictionary with equivalence img_id <->> classe
        self.image_to_classes = {}
        self.__build_image_to_classes()

    def __getitem__(self, index: int):
        img_id = list(self.img_annotations.keys())[index]
        img_name = self.root_dir / img_id

        if self.augmentations:
            image = self.augmentations(image=io.imread(img_name))["image"]
        else:
            image = Image.fromarray(io.imread(img_name))

        if self.image_to_classes:
            return image, self.image_to_classes[img_id]
        else:
            raise NotImplementedError

    def __len__(self) -> int:
        return len(self.img_annotations.keys())

    def __get_label_for_id(self, label_id: str) -> str:
        """
        Method to get the label from a label id using the label mapping

        Argument:
        ---------
            - label_id (str): id of the label
        Return:
        -------
            - label (str)
        """
        return self.csv_mapping[self.csv_mapping[Constants.COL_LABEL_ID] == label_id][
            Constants.COL_LABEL_NAME + self.__lang
        ].values[0]

    def __is_aliment_present(self, image_id: str) -> bool:
        """
        Method to check if an aliment is present in an image.

        Argument:
        ---------
            - image_id (str): id of the image we want to check
        Return:
        -------
            - boolean: true if the image contains aliment, false else
        """
        func = lambda dict_box: self.__get_label_for_id(dict_box["id"])
        r = re.compile(self.__regex_aliment)
        if any(r.match(w) for w in map(func, self.img_annotations[image_id])):
            return True
        return False

    def __build_image_to_classes(self):
        """
        Method to build the dictionary image <-> classes
        """
        for img_id in self.img_annotations.keys():
            if self.__is_aliment_present(img_id):
                self.image_to_classes[img_id] = 1
            else:
                self.image_to_classes[img_id] = 0


def split_train_test_valid_json(img_annotation_path: str, random_seed: int = None, split_size=(0.8, 0.2)):
    """
    Method to split dataset ids into train, test and valid

    Argument:
    ---------
        - img_annotation_path (str): filename/path of the annotation json file
        - random_seed (int): seed of the random shuffle
        - split_size (tuple of float): float size for each split

    Return:
    -------
        - 2 or 3 dictionary corresponding to the splitted annotation json file
    """
    with open(img_annotation_path) as f:
        img_annotation = json.load(f)

    img_ids = list(img_annotation.keys())
    if random_seed:
        random.seed(random_seed)
    random.shuffle(img_ids)
    total_length = len(img_ids)
    img_ids = np.array(img_ids)

    if len(split_size) == 1 and split_size[0] <= 1:
        split_key = np.split(img_ids, [np.floor(total_length * split_size[0])])
        return (
            {k: v for k, v in img_annotation.items() if k in split_key[0]},
            {k: v for k, v in img_annotation.items() if k in split_key[1]},
        )
    elif len(split_size) == 2 and split_size[0] <= 1:
        split_key = np.split(img_ids, [int(np.floor(total_length * split_size[0]))])
        return (
            {k: v for k, v in img_annotation.items() if k in split_key[0]},
            {k: v for k, v in img_annotation.items() if k in split_key[1]},
        )
    elif len(split_size) == 3 and split_size[0] <= 1:
        split_key = np.split(
            img_ids,
            [
                int(np.floor(total_length * split_size[0])),
                int(np.floor(total_length * (split_size[0] + split_size[1]))),
            ],
        )
        return (
            {k: v for k, v in img_annotation.items() if k in split_key[0]},
            {k: v for k, v in img_annotation.items() if k in split_key[1]},
            {k: v for k, v in img_annotation.items() if k in split_key[2]},
        )

    else:
        raise NotImplementedError