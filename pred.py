from pathlib import Path
import glob
import albumentations as A
import matplotlib.pyplot as plt
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import ImageFolder
from skimage import io
import numpy as np

from src.classifier import predict_img, load_checkpoint, get_cam
from src.utils import init_parser_pred


def load_transformation_pipelines():
    # Define the transformation pipeline for test
    tranformation_pipeline_test = A.Compose(
        [
            A.Resize(width=512, height=512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),  # convert the image to PyTorch tensor
        ],
        p=1,
    )

    return tranformation_pipeline_test


def plot_result(image_path, output_prob):
    fig, ax = plt.subplots()
    img = Image.open(image_path)
    if output_prob < 0.5:
        title = "Image contains tomato ({:02.2f})".format(output_prob, 4)
    else:
        title = "Image do not contains tomato ({:02.2f})".format(output_prob, 4)

    ax.imshow(img)
    ax.set_title(title)
    plt.show()


if __name__ == "__main__":
    # Init parser ARGS
    print("Init parser ..")
    ARGS = init_parser_pred()
    print("Retrieve args ..")
    MODEL_PATH = Path(ARGS.model_filepath)
    IMG_PATH = Path(ARGS.img_filepath)
    if IMG_PATH.is_file():
        #IMG_PATH = Path("assignment_imgs/6e23ca89ab13945cbfa3efc0a8e406f4.jpeg")
        #IMG_PATH = Path("assignment_imgs/6d5427671eb3c668320cec72f83ef573.jpeg")

        # Define the transformation pipeline for test
        tranformation_pipeline_test = load_transformation_pipelines()

        print("Build classifier & model ..")

        model = load_checkpoint(MODEL_PATH)
        # Build classifier
        output = predict_img(model, io.imread(IMG_PATH), tranformation_pipeline_test)
        print(output.data.cpu().numpy()[0][0])

        if ARGS.plot_result:
            plot_result(IMG_PATH, output.data.cpu().numpy()[0][0])

        #test = get_cam(model, "layer4", IMG_PATH, tranformation_pipeline_test)
    else:
        # Define the transformation pipeline for test
        tranformation_pipeline_test = load_transformation_pipelines()
        print("Build classifier & model ..")
        model = load_checkpoint(MODEL_PATH)

        img_list = glob.glob(IMG_PATH.as_posix()+'/*.jpeg')
        for img in img_list:
            print(img)
            output = predict_img(model, io.imread(img), tranformation_pipeline_test)
            print(output.data.cpu().numpy()[0][0])
            print("--------------")

