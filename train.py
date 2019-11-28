from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from src.classifier import AlimentClassifier
from src.foodvisor_dataset import (
    FoodVisorDataset,
    split_train_test_valid_json,
)
from src.utils import init_parser_train, parse_config_file


def load_agumentation_pipelines():
    # Define the augmentation pipeline
    augmentation_pipeline_train = A.Compose(
        [
            A.Resize(width=512, height=512),
            A.HorizontalFlip(p=0.5),  # apply horizontal flip to 50% of images
            A.Rotate(
                limit=90, p=0.5
            ),  # apply random with limit of 90° to 50% of images
            A.OneOf(
                [
                    # apply one of transforms to 30% of images
                    A.RandomBrightnessContrast(),  # apply random contrast & brightness
                    A.RandomGamma(),  # apply random gamma
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    # apply one of transforms to 30% images
                    A.ElasticTransform(
                        alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
                    ),
                    A.GridDistortion(),
                    A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                ],
                p=0.3,
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),  # convert the image to PyTorch tensor
        ],
        p=1,
    )

    # Define the transformation pipeline for test
    tranformation_pipeline_test = A.Compose(
        [
            A.Resize(width=512, height=512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),  # convert the image to PyTorch tensor
        ],
        p=1,
    )

    return augmentation_pipeline_train, tranformation_pipeline_test


if __name__ == "__main__":
    # Init parser ARGS
    print("Init parser ..")
    ARGS = init_parser_train()
    print("Retrieve args ..")
    FOLDER_IMGS = Path(ARGS.folder_imgs)
    IMG_ANNOTATIONS_PATH = Path(ARGS.img_annotations)
    LABEL_MAPPING_PATH = Path(ARGS.label_mapping)
    MODEL_FILEPATH = Path(ARGS.model_filepath)
    CONFIG_PATH = Path(ARGS.config)

    # Load param dictionnary
    params_loader, params_classifier, param_common = parse_config_file(CONFIG_PATH)

    print("Set up datasets ..")
    # Load augmentation pipelines
    augmentation_pipeline_train = load_agumentation_pipelines()[0]
    tranformation_pipeline_test = load_agumentation_pipelines()[1]

    (
        img_annotations_train,
        img_annotations_test,
        img_annotations_valid,
    ) = split_train_test_valid_json(
        IMG_ANNOTATIONS_PATH, random_seed=42, split_size=param_common["split_size_train_test"]
    )

    # Build dataset
    food_dataset_train = FoodVisorDataset(
        json_annotations=img_annotations_train,
        csv_mapping=LABEL_MAPPING_PATH,
        root_dir=FOLDER_IMGS,
        regex_aliment=param_common["regex_aliment"],
        augmentations=augmentation_pipeline_train,
    )
    food_dataset_test = FoodVisorDataset(
        json_annotations=img_annotations_test,
        csv_mapping=LABEL_MAPPING_PATH,
        root_dir=FOLDER_IMGS,
        regex_aliment=param_common["regex_aliment"],
        augmentations=tranformation_pipeline_test,
    )
    food_dataset_valid = FoodVisorDataset(
        json_annotations=img_annotations_valid,
        csv_mapping=LABEL_MAPPING_PATH,
        root_dir=FOLDER_IMGS,
        regex_aliment=param_common["regex_aliment"],
        augmentations=tranformation_pipeline_test,
    )

    # Build train and test loader
    train_loader = DataLoader(
        food_dataset_train,
        batch_size=params_loader["batch_size"],
        shuffle=params_loader["shuffle_dataset"],
    )
    test_loader = DataLoader(
        food_dataset_test,
        batch_size=params_loader["batch_size"],
        shuffle=params_loader["shuffle_dataset"],
    )
    valid_loader = DataLoader(
        food_dataset_valid,
        batch_size=params_loader["batch_size"],
        shuffle=params_loader["shuffle_dataset"],
    )

    print("Build classifier & model ..")
    # Build classifier
    classifier = AlimentClassifier()
    classifier.build_model()

    print("Start training ..")
    # Train
    classifier.train_classifier(
        train_loader=train_loader,
        test_loader=test_loader,
        params=params_classifier,
        livelossplot=False,
        save_checkpoint_each=param_common["save_checkpoint_each"],
    )
