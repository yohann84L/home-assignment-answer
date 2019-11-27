import time

import torch
from livelossplot import PlotLosses
from torchvision import models
from skimage import io

from src.utils import Constants


class AlimentClassifier:
    """
    Custom class to train, test and make inference for binary classification of aliment in a picture.

    Arguments:
    ----------
        - optimizer (default = Adam): optimizer of the classifier
        - loss (default = BCELoss): loss function of the classifier
        - params_classifier (dict): parameters for the dataset loader, params are:
                "threshold": 0.5,
                "epochs": 40,
                "learning_rate": 0.001
    """

    def __init__(self, optimizer=torch.optim.Adam, loss=None, params_classifier=None):
        # Define some variables
        self.model = None
        self.optimizer = optimizer
        self.logs = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Loss function and gradient descent
        if loss is None:
            # weights = [500 / 3000]
            # class_weights = torch.tensor(weights, dtype=torch.float, device=self.device)
            self.loss = torch.nn.BCELoss()
        else:
            self.loss = loss

        # Parameters
        if params_classifier is None:
            self.params_classifier = Constants.DEFAULT_PARAM_CLASSIFIER
        else:
            self.params_classifier = params_classifier

    def build_model(
        self,
        # model_pretrained=models.mobilenet_v2(pretrained=True),
        model_pretrained=None,
        feature_extract=True,
    ):
        """
        Method to build the model: pretrained model with new top layer classifier.

        Arguments:
        ----------
            - model_pretrained : pretrained model we want to use, currently only resnet50 is supported,
                if changed, top layer classifier may be broke
            - feature_extract (bool=True): if true, only top layers classifier
                parameters will be updated
        """
        # Build model using pretrained model for transfer learning
        if model_pretrained is None:
            self.model = models.resnet50(pretrained=True)
        else:
            self.model = model_pretrained

        # Fix pretrenained model if feature extract, no need to backpropagate
        # But if we're finetuning we do not fix
        self.set_parameter_requires_grad(self.model, feature_extract)

        # Build top layer for classification (binary here)
        num_ftrs = self.model.fc.in_features
        self.model.fc = TopLayerResNet50(in_size=num_ftrs, out_size=1)

        # Get the parameters of the model to update during training,
        # it depends of which parameters requires grad
        params_to_update = self.model.parameters()
        print("Params to learn:")
        if feature_extract:
            params_to_update = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    params_to_update.append(param)
                    print("\t", name)
        else:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print("\t", name)
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
            - params (dict) : if needed to update some paramters such as epochs without rebuilding
            the entire class put the updated parameters here
            - livelossplot (bool=False): use livelossplot to plot running loss and error_rate
            - save_checkpoint_each (list): list of epoch when we want to save model
        """
        # Update parameters if given

        if save_checkpoint_each is None:
            save_checkpoint_each = self.params_classifier["epochs"] - 1
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

                for images, labels in iter(loader_dict[phase]):
                    images = images.to(self.device)
                    labels = torch.tensor(labels, dtype=torch.float, device=self.device)

                    # Compute forward
                    output = self.model.forward(images)
                    loss = self.loss(output, labels)

                    # Do the retropropag if in train phase
                    if phase == "train":
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    # Compute prediction
                    preds = torch.where(
                        output >= self.params_classifier["threshold"],
                        torch.tensor(1, device=self.device),
                        torch.tensor(0, device=self.device),
                    )
                    running_loss += loss.detach() * images.size(0)
                    running_uncorrects += torch.sum(
                        preds != labels.data.detach().view(labels.data.size()[0], 1)
                    )

                # Update loss and error_rate for the last epoch
                if phase == "train":
                    size_loader = loader_dict[phase].sampler.num_samples
                else:
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
                string_print = ""

            # Save checkpoint
            if (e + 1) in save_checkpoint_each:
                save_checkpoint(
                    self.model, model_name="ResNet50_checkpoint_e{}.pth".format(e)
                )

        # Print training time
        time_elapsed = time.time() - since
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

    @staticmethod
    def set_parameter_requires_grad(model, feature_extracting: bool):
        """
        Static method to set parameters requires grad or not. It depends on if we want to train
        all paramters or only top layers classifier

        Arguments:
        ----------
            - model
            - feature_extracting (boolean): if true, only top layers classifier
             parameters will be updated
        """
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False


def save_checkpoint(model, model_name="checkpoint.pth"):
    """
    Function to save model
    """
    checkpoint = {
        "arch": "ResNet50+TopLayerResNet50",
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, model_name)


def load_checkpoint(filepath):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(filepath, map_location=torch.device(device))
    if checkpoint['arch'] == 'ResNet50+TopLayerResNet50':
        model = models.resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
    else:
        print("Architecture not recognized.")
        raise ValueError

    # Fix pretrenained model if feature extract, no need to backpropagate
    # But if we're finetuning we do not fix
    AlimentClassifier.set_parameter_requires_grad(model, True)

    # Build top layer for classification (binary here)
    num_ftrs = model.fc.in_features
    model.fc = TopLayerResNet50(in_size=num_ftrs, out_size=1)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def predict_img(model, img, transformation_pipeline):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = transformation_pipeline(image=img)['image']
    image = image.unsqueeze(0).to(device)
    output = model.forward(image)
    return output

def get_cam(model, last_conv_layer, image_path, transformation_pipeline):
    import numpy as np
    from torch.autograd import Variable
    import matplotlib.pyplot as plt

    conv_fmap = []
    def hook(module, input, output):
        return conv_fmap.append(output.data.cpu().numpy())

    model._modules.get(last_conv_layer).register_forward_hook(hook)

    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-4].data.cpu().numpy())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = transformation_pipeline(image=io.imread(image_path))['image']
    image = Variable(image.unsqueeze(0)).to(device)

    logit = model(image)

    h_x = torch.nn.functional.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    bz, nc, h, w = conv_fmap[0].shape
    output_cam = []

    print(bz, nc, h, w)
    print(weight_softmax.shape)

    cam = weight_softmax.dot(conv_fmap[0].reshape((nc, h*w)))
    cam = cam.reshape(h*w, h*w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    plt.imshow(cam_img)
    plt.show()



class TopLayerResNet50(torch.nn.Module):
    """
    Modified classifier for ResNet50 in order to adapt the model to our problem (binary classification).

    Arguments:
    ----------
        - in_size (int=2048): input size of last layer of ResNet50, it should be 2048
        - out_size (int=1): number of classes
    """

    def __init__(self, in_size: int = 2048, out_size: int = 1):
        super(TopLayerResNet50, self).__init__()
        self.linear1 = torch.nn.Linear(in_features=in_size, out_features=256, bias=True)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.linear2 = torch.nn.Linear(
            in_features=256, out_features=out_size, bias=True
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x
