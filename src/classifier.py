import time

import torch
from livelossplot import PlotLosses
from skimage import io
from torchvision import models

from src.utils import Constants


class AlimentClassifier:
    """
    Custom class to train, test and make inference for binary classification of aliment in a picture.

    Arguments:
    ----------
        - optimizer (default = Adam): optimizer of the classifier
        - loss (default = CrossEntropyLoss): loss function of the classifier
        - loss_weight (list): weight for each class if unbalanced
        - params_classifier (dict): parameters for the dataset loader, params are:
                "threshold": 0.5,
                "epochs": 40,
                "learning_rate": 0.001
    """

    def __init__(self, params_classifier, optimizer=torch.optim.Adam, loss=None):
        # Define some variables
        self.model = None
        self.optimizer = optimizer
        self.logs = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Loss function and gradient descent
        if loss is None:
            if "loss_weight" in params_classifier.keys():
                # Because class are unbalance we define a weightfor each class
                class_weights = torch.tensor(
                    params_classifier["loss_weight"],
                    dtype=torch.float,
                    device=self.device,
                )
                self.loss = torch.nn.CrossEntropyLoss(weight=class_weights)
            else:
                self.loss = torch.nn.CrossEntropyLoss()
        else:
            self.loss = loss

    def build_model(self, model=models.resnet18, pretrained=True):
        """
        Method to build the model: pretrained model with new top layer classifier.

        Arguments:
        ----------
            - model : model we want to use, currently only resnet18 is supported,
                if changed, top layer classifier may be broke
            - pretrained (boolean): use or not the pretrained model
        """
        self.model = model(pretrained=pretrained)

        # Freeze 6 first layers
        count = 0
        params_to_update = []
        for child in self.model.children():
            count += 1
            if count < 7:
                for param in child.parameters():
                    param.requires_grad = False

        # Update last layers ouputs
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, 2)

        # Print parameters we'll learn
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print("\t", name)

        # Define optimizer
        self.optimizer = self.optimizer(
            params_to_update, lr=self.params_classifier["learning_rate"]
        )

    def train_classifier(
        self,
        train_loader,
        test_loader,
        params: dict = None,
        livelossplot=False,
        save_checkpoint_each=None,
    ):
        """
        Method to train the model.

        Arguments:
        ----------
            - train_loader : DatasetLoader for the training set
            - test_loader : DatasetLoader for the test set
            - params (dict) : if needed to update some parameters such as epochs without rebuilding
            the entire class put the updated parameters here
            - livelossplot (bool=False): use livelossplot to plot running loss and error_rate
            - save_checkpoint_each (list): list of epoch when we want to save model
        """
        # Update parameters if given

        if save_checkpoint_each is None:
            save_checkpoint_each = [self.params_classifier["epochs"]]
        if params:
            for param, value in params.items():
                self.params_classifier[param] = value

        # Define liveloss and time of training start
        if livelossplot:
            liveloss = PlotLosses()
        since = time.time()

        # Show which device is used
        print("Using device {}".format(self.device))
        self.model.to(self.device)

        loader_dict = {"train": train_loader, "validation": test_loader}
        for e in range(self.params_classifier["epochs"]):
            self.logs = {}
            if not livelossplot:
                print("Epoch {}/{} :".format(e, self.params_classifier["epochs"]))
                print("--------------")
            # Alternate between train and validation phase
            for phase in ["train", "validation"]:
                if phase == "train":
                    self.model.train()
                else:
                    self.model.eval()

                # Define loss and uncorrects predictions
                running_loss = 0.0
                running_uncorrects = 0

                # Loop over loader
                for images, labels in iter(loader_dict[phase]):
                    images = images.to(self.device)
                    labels = torch.tensor(labels, dtype=torch.long, device=self.device)

                    # Compute forward
                    output = self.model.forward(images)
                    loss = self.loss(output, labels)

                    # Do the retropropag if in train phase
                    if phase == "train":
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    # Compute prediction
                    _, predicted = torch.max(output, 1)
                    running_loss += loss.detach() * images.size(0)
                    running_uncorrects += torch.sum(predicted != labels.data.detach())

                # Compute loss and error_rate
                size_loader = len(loader_dict[phase].dataset)
                epoch_loss = running_loss / size_loader
                epoch_error_rate = running_uncorrects.float() / size_loader

                # Set the prefix for logs
                prefix = ""
                if phase == "validation":
                    prefix = "val_"

                # Update logs
                self.logs[prefix + "log loss"] = epoch_loss.item()
                self.logs[prefix + "error_rate"] = epoch_error_rate.item()

            # Use liveloss to plot loss and accuracy
            if livelossplot:
                liveloss.update(self.logs)
                liveloss.draw()
            else:
                string_print = """
                Training:               |   Validation:
                    log loss = {}       |       val_log loss = {}
                    error_rate = {}     |       val_error_rate = {}
                """.format(
                    self.logs["log loss"], self.logs["val_log loss"],
                    self.logs["error_rate"], self.logs["val_error_rate"]
                )
                print(string_print)

            # Save checkpoint
            if (e + 1) in save_checkpoint_each:
                save_checkpoint(
                    self.model, model_name="AlexNet_checkpoint_e{}.pth".format(e)
                )

        # Print training time
        time_elapsed = time.time() - since
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

    @staticmethod
    def evaluate(valid_loader, model, device="cuda:0"):
        """
        Method to evaluate a test set.

        Argument:
        ---------
            - valid_loader (dataloader): loader with dataset to evaluate
            - model (pytorch model): model to evaluate
            - device (str="cuda:0"): device to use
        """
        # Set model into eval mode
        model.eval()

        # Iterate over the valid_dataset
        pred_proba = torch.empty(0, 2, device=device).zero_().float()
        true_labels = torch.empty(0, device=device).zero_().float()
        for images, labels in valid_loader:
            images = images.to(device)
            labels = torch.tensor(
                labels.clone().detach(), dtype=torch.float, device=device
            )
            # Append true labels
            true_labels = torch.cat((true_labels, labels))

            # Predict and appends predicted values
            output = model.forward(images)
            pred_proba = torch.cat((pred_proba, output))

        return pred_proba, true_labels


def save_checkpoint(model, model_name="checkpoint.pth"):
    """
    Function to save model
    """
    checkpoint = {
        "arch": "ResNet18",
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, model_name)


def load_checkpoint(filepath, device="cpu"):
    """
    Function to load model
    """
    checkpoint = torch.load(filepath, map_location=device)
    if checkpoint["arch"] == "ResNet18":
        model = models.resnet18(pretrained=True)
        # Freeze 6 first layers
        count = 0
        params_to_update = []
        for child in model.children():
            count += 1
            if count < 7:
                for param in child.parameters():
                    param.requires_grad = False

        # Update last layers ouputs
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 2)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
    else:
        print("Architecture not recognized.")
        raise ValueError





def predict_img(model, img, transformation_pipeline):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = transformation_pipeline(image=img)["image"]
    image = image.unsqueeze(0).to(device)
    model.eval()
    output = model.forward(image)
    return output


def get_cam(model, last_conv_layer, image_path, transformation_pipeline):
    """
    Method under construction
    """
    import numpy as np
    from torch.autograd import Variable
    import matplotlib.pyplot as plt
    from PIL import Image

    conv_fmap = []

    def hook(module, input, output):
        return conv_fmap.append(output.data.cpu().numpy())

    model._modules.get(last_conv_layer).register_forward_hook(hook)

    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = transformation_pipeline(image=io.imread(image_path))["image"]
    image = Variable(image.unsqueeze(0)).to(device)

    logit = model(image)

    bz, nc, h, w = conv_fmap[0].shape

    print(bz, nc, h, w)
    print(weight_softmax.shape)

    cam = weight_softmax.dot(conv_fmap[0].reshape((nc, h * w)))
    print(cam.shape)
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = Image.fromarray(np.uint8(255 * cam_img))
    base_img = Image.open(image_path)
    base_img.resize((h, w))
    cam_img.show()
    base_img.paste(cam_img, (0, 0))
    base_img.show()
    # plt.imshow(base_img)
    # plt.show()
