import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage import io
from torchvision.transforms.functional import to_pil_image


class ClassActivationMapping:
    def __init__(
        self,
        model,
        image_path,
        transformation_pipeline,
        last_conv_layer="layer4",
        device="cpu",
    ):
        self.model = model

        self.grad_maps = []

        self.model._modules[last_conv_layer].register_forward_hook(self.__register_grad)
        self.weight_softmax = self.model.fc.weight.to(device)
        self.image = io.imread(image_path)
        self.__forward_img(transformation_pipeline, device)

    def __register_grad(self, module, input, output):
        self.grad_maps.append(output.data)
        return None

    def __forward_img(self, transformation_pipeline, device):
        image = transformation_pipeline(image=self.image)["image"]
        image = torch.tensor(image.unsqueeze(0)).to(device)
        self.model.forward(image)

    def get_cam(self):
        cam = self.weight_softmax[1, :] @ torch.flatten(self.grad_maps[0], 2)
        cam = cam.view(
            self.grad_maps[0].size(0),
            1,
            self.grad_maps[0].size(3),
            self.grad_maps[0].size(2),
        )
        cam = to_pil_image(cam[0, 0].detach().numpy())
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        cam = Image.fromarray(np.uint8(255 * cam))
        cam = cam.resize(self.image.shape[0:2], resample=Image.BICUBIC)
        return cam

    def show_cam_on_image(self, cam):
        cam_color = cv2.applyColorMap(np.uint8(255 * np.array(cam)), cv2.COLORMAP_JET)
        img_cam = np.float32(cam_color) + np.float32(self.image)
        img_cam = img_cam / np.max(img_cam)
        plt.imshow(img_cam)
        plt.show()
