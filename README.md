# Home assignment answer - Tomatoes Allergies
---
<a href="https://www.python.org/"><img alt="Python Version" src="https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7-blue"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

---
You will find in this repository code of the [FoodVisor Home Assigment](https://github.com/Foodvisor/home-assignment).

Architecture of folder/file:
```
.
└── home-assigment-answer
```

## Prerequisites
To build this classifier I used PyTorch. I trained the model on using Google Colab. To install required packages:
```
pip install -r requirements.txt
```
## Run Training
To train the model use:
```
python train.py folder_imgs img_annotations label_mapping model_filepath config
```

With:
- `folder_imgs`: path folder with image dataset
- `img_annotations`: path file for `the annotations.json`
- `label_mapping`: path file for the `label_mapping.csv`
- `model_filepath`: path file to save the model checkpoint
- `config`: path file for the `config.json` file. 
Default `config.json`:
```json
{
    "param_loader": {
        "batch_size": 32,
        "validation_split": 0.2,
        "shuffle_dataset": true,
        "random_seed": 42
    },
    "param_classifier": {
        "threshold": 0.5,
        "learning_rate": 0.001,
        "epochs": 40
    }
}
```

## Run Prediction
To predict class from an image use:
```
python pred.py model_filepath img_filepath
```
With:
- `model_filepath`: filepath for the model checkpoint
- `img_filepath`: filepath for the image we want to predict class


## Classifation and Results
I will briefly explain here what've done so far and the results I've got.

### Problematic
The goal of the assignment is to build a classifier in order to 
know if a picture (of a meal) contains tomatoes. Here by "contains 
tomatoes" we mean "fresh tomatoes". Thus, we'll not detect Ketchup for example.

### Approach
The given dataset is composed of 3000 images, each image has an associated dictionnary 
containing bounding box of aliment present in the meal.
Here, two options comes-up:
1. build a binary classification for each image
2. use the bouding box and an object detector to detect tomatoes

I will follow the first way as it should give almost the same results and is less 
complicated to build.

### Classifier
Several options are available here to build a classifier:
1. build image classifier from scratch using CNN for example
2. use well know classifier architecture from scratch (ResNet, Alexnet, VGG, Inception etc..)
3. use transfer learning with pretrained architecture (same ones)

Considering the number of sample we have, it is more interesting to use transfer learning
to use the pretrained feature extractor and avoid a bit of overfitting. As we do not have a big dataset and our data is not 
similar to ImageNet (the dataset used to train ResNet), we'll freeze the only some lower layer. I choose to freeze the first 6
layer (this number is choosen quite randomly) and change the last layer to match out 2 ouput class.

PyTorch offer a lot of pretrained model in `torchvision.models`, I will use here ResNet18 ([Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)).
This model is very lighteweight compared to ResNet50 or 102, it is good choice to get first results.


Workflow to train the classifier:
- Build dataset using `FoodVisorDataset`
- Build loader using pytorch `DatasetLoader`, we use a sampler for the loader:
- Build model classifier with `AlimentClassifier().build_model()`
- Train model with `AlimentClassifier().train_classifier(train_loader, test_loader)`

### Results

Here are the plot of training and test error_rate:

UPCOMING IN THE AFTERNOON

### Model Release
ResNet18 with 6 first layers freezed, trained on 1950 images : UPCOMING IN THE AFTERNOON



## References
- [FoodVisor Home Assigment](https://github.com/Foodvisor/home-assignment)
- [Custom Datasets/Dataloaders in PyTorch](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#writing-custom-datasets-dataloaders-and-transforms)
- [Example custom dataset](https://github.com/utkuozbulak/pytorch-custom-dataset-examples#custom-dataset-fundamentals)
- [Example PyTorch Image Classification](https://github.com/LeanManager/PyTorch_Image_Classifier)
- [Albumentations library](https://github.com/albumentations-team/albumentations)
- [Livelossplot librairie](https://github.com/stared/livelossplot)
- [Save and loading checkpoint function from](https://github.com/LeanManager/PyTorch_Image_Classifier/blob/master/Image_Classifier_Project.ipynb)