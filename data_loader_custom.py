import torch
import cv2
# data loader
import glob
import torch
import torchvision.ops.misc
from torchvision import transforms
from skimage import io, transform, color
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2


class SalObjDatasetNew(Dataset):
    def __init__(self, img_name_list, lbl_name_list, size_wh=(1280, 640)):
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.transform_image = transforms.Compose(
            [
                transforms.RandomApply([
                    # transforms.RandomCrop(size=(1200, 2400)),
                    transforms.RandomRotation(10, expand=True, resample=Image.BILINEAR),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip()],
                    p=0.4),
                transforms.RandomApply([
                    transforms.ColorJitter(contrast=0.3, hue=0.5)
                    ],
                    p=0.4),
                transforms.Resize(size=(size_wh[1], size_wh[0]), interpolation=Image.BILINEAR),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.transform_label = transforms.Compose(
            [
                transforms.RandomApply([
                    # transforms.RandomCrop(size=(1200, 2400)),
                    transforms.RandomRotation(10, expand=True, resample=Image.NEAREST),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip()],
                    p=0.4),
                transforms.Resize(size=(size_wh[1], size_wh[0]), interpolation=Image.NEAREST)
            ]
        )

    def __getitem__(self, idx):
        image = cv2.imread(self.image_name_list[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # to rgb
        if len(self.label_name_list) == 0:
            label = np.zeros(image.shape[0:2])
            label = np.expand_dims(label, -1)
        else:
            label = cv2.imread(self.label_name_list[idx], -1)
            label = np.expand_dims(label, -1)
        image = np.transpose(image, (2, 0, 1))
        label = np.transpose(label, (2, 0, 1))
        image = np.asarray(image,  dtype=np.float32) / 255.0
        label = np.asarray(label,  dtype=np.float32)
        image_tensor = torch.from_numpy(image)
        label_tensor = torch.from_numpy(label)

        seed = np.random.randint(-65534, 65534)
        torch.manual_seed(seed)
        image_tensor = self.transform_image(image_tensor)
        torch.manual_seed(seed)
        label_tensor = self.transform_label(label_tensor)

        return image_tensor, label_tensor

    def __len__(self):
        return len(self.image_name_list)



if __name__ == "__main__":
    import random

    image1 = cv2.imread('test/1.jpg')
    label1 = cv2.imread('test/1.png', -1)
    label1 = np.expand_dims(label1, -1)

    image1 = np.transpose(image1, (2, 0, 1))
    label1 = np.transpose(label1, (2, 0, 1))

    image1 = torch.from_numpy(image1)
    label1 = torch.from_numpy(label1)
    for i in range(10):
        seed = np.random.randint(-65534, 65534)
        t1 = transforms.Resize(size=(640, 1280), interpolation=Image.BILINEAR)
        # t2 = transforms.RandomCrop(size=(1200, 2400))
        t2 = transforms.RandomApply([
            # transforms.RandomCrop(size=(1400, 2800)),
            transforms.RandomRotation(15, expand=True, resample=Image.NEAREST),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.ColorJitter(contrast=0.3, hue=0.5)
        ],
            p=0.8
        )
        transform_image = transforms.Compose([t2, t1])
        torch.manual_seed(seed)
        image1_output = transform_image(image1)
        torch.manual_seed(seed)
        label1_output = transform_image(label1)
        img_out = image1_output.data.numpy()
        lab_out = label1_output.data.numpy()
        img_out = np.transpose(img_out, (1, 2, 0))
        lab_out = np.transpose(lab_out, (1, 2, 0))

        cv2.imshow("test_img".format(i), img_out)
        cv2.imshow("test_label".format(i), lab_out*255)
        cv2.waitKey()

