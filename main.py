from torchvision import transforms
import albumentations as A
from pathlib import Path
from albumentations.pytorch import ToTensorV2
from src.foodvisor_dataset import FoodVisorDataset, split_train_test_valid_json
from src.classifier import AlimentClassifier, load_checkpoint, get_cam
import torch

def main():
    # Define the augmentation pipeline
    augmentation_pipeline_train = A.Compose(
        [
            A.Resize(width=512, height=512),
            A.HorizontalFlip(p=0.5),  # apply horizontal flip to 50% of images
            A.Rotate(limit=90, p=0.5), # apply random with limit of 90° to 50% of images
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

            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),

            ToTensorV2()  # convert the image to PyTorch tensor
        ],
        p=1,
    )

    # Define the transformation pipeline for test
    tranformation_pipeline_test = A.Compose(
        [
            A.Resize(width=512, height=512),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            ToTensorV2()  # convert the image to PyTorch tensor
        ],
        p=1,
    )

    IMG_ANNOTATIONS_PATH = Path("references/img_annotations.json")
    LABEL_MAPPING_PATH = Path("references/label_mapping.csv")
    FOLDER_IMGS = Path("assignment_imgs/")
    (
        img_annotations_train,
        img_annotations_test,
        img_annotations_valid,
    ) = split_train_test_valid_json(
        IMG_ANNOTATIONS_PATH, random_seed=42, split_size=[0.65, 0.25, 0.1]
    )

    # Build dataset
    food_dataset_train = FoodVisorDataset(
        json_annotations=img_annotations_train,
        csv_mapping=LABEL_MAPPING_PATH,
        root_dir=FOLDER_IMGS,
        regex_aliment=r"[Tt]omate(s)?",
        augmentations=augmentation_pipeline_train,
    )
    food_dataset_test = FoodVisorDataset(
        json_annotations=img_annotations_test,
        csv_mapping=LABEL_MAPPING_PATH,
        root_dir=FOLDER_IMGS,
        regex_aliment=r"[Tt]omate(s)?",
        augmentations=tranformation_pipeline_test,
    )
    food_dataset_valid = FoodVisorDataset(
        json_annotations=img_annotations_valid,
        csv_mapping=LABEL_MAPPING_PATH,
        root_dir=FOLDER_IMGS,
        regex_aliment=r"[Tt]omate(s)?",
        augmentations=tranformation_pipeline_test,
    )

    params_loader = {"batch_size": 32,
                     "validation_split": 0.2,
                     "shuffle_dataset": True,
                     "random_seed": 42}

    # Init classifier with food_dataset
    #loaders = FoodDatasetLoader(food_dataset_train, food_dataset_test, param_loader=params_loader)

    # Build train and test loader
    #train_loader, test_loader = loaders.build_loader()

    # Build classifier
    # classifier = AlimentClassifier()
    # classifier.build_model()

    # Train
    # classifier.train_classifier(train_loader, test_loader)



if __name__ == "__main__":
    # Define the augmentation pipeline
    augmentation_pipeline_train = A.Compose(
        [
            A.Resize(width=512, height=512),
            A.HorizontalFlip(p=0.5),  # apply horizontal flip to 50% of images
            A.Rotate(limit=90, p=0.5), # apply random with limit of 90° to 50% of images
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

            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),

            ToTensorV2()  # convert the image to PyTorch tensor
        ],
        p=1,
    )

    # Define the transformation pipeline for test
    tranformation_pipeline_test = A.Compose(
        [
            A.Resize(width=512, height=512),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            ToTensorV2()  # convert the image to PyTorch tensor
        ],
        p=1,
    )

    IMG_ANNOTATIONS_PATH = Path("references/img_annotations.json")
    LABEL_MAPPING_PATH = Path("references/label_mapping.csv")
    FOLDER_IMGS = Path("assignment_imgs/")
    (
        img_annotations_train,
        img_annotations_test,
        img_annotations_valid,
    ) = split_train_test_valid_json(
        IMG_ANNOTATIONS_PATH, random_seed=42, split_size=[0.65, 0.25, 0.1]
    )

    # Build dataset
    food_dataset_train = FoodVisorDataset(
        json_annotations=img_annotations_train,
        csv_mapping=LABEL_MAPPING_PATH,
        root_dir=FOLDER_IMGS,
        regex_aliment=r"[Tt]omate(s)?",
        augmentations=augmentation_pipeline_train,
    )
    food_dataset_test = FoodVisorDataset(
        json_annotations=img_annotations_test,
        csv_mapping=LABEL_MAPPING_PATH,
        root_dir=FOLDER_IMGS,
        regex_aliment=r"[Tt]omate(s)?",
        augmentations=tranformation_pipeline_test,
    )
    food_dataset_valid = FoodVisorDataset(
        json_annotations=img_annotations_valid,
        csv_mapping=LABEL_MAPPING_PATH,
        root_dir=FOLDER_IMGS,
        regex_aliment=r"[Tt]omate(s)?",
        augmentations=tranformation_pipeline_test,
    )

    params_loader = {"batch_size": 32,
                     "validation_split": 0.2,
                     "shuffle_dataset": True,
                     "random_seed": 42}

    # Init classifier with food_dataset
    # loaders = FoodDatasetLoader(food_dataset_train, food_dataset_test, weights, param_loader=params_loader)

    # Build train and test loader
    train_loader = torch.utils.data.DataLoader(food_dataset_train, batch_size=params_loader["batch_size"],
                                               shuffle=params_loader["shuffle_dataset"])
    test_loader = torch.utils.data.DataLoader(food_dataset_test, batch_size=params_loader["batch_size"],
                                              shuffle=params_loader["shuffle_dataset"])
    valid_loader = torch.utils.data.DataLoader(food_dataset_valid, batch_size=params_loader["batch_size"],
                                               shuffle=False)

    model = load_checkpoint("models/ResNet50_checkpoint_e40.pth", device="cpu")
    y_pred, y_true = AlimentClassifier.evaluate(valid_loader, model, device="cpu")

    #get_cam(model, "layer4", "image_to_predict/6e23ca89ab13945cbfa3efc0a8e406f4.jpeg", tranformation_pipeline_test)