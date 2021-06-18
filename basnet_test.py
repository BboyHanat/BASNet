import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # , utils
# import torch.optim as optim
import cv2
import numpy as np
from PIL import Image
import glob

from data_loader_custom import SalObjDatasetTest
from model import BASNet


def draw_mask(image, masks, plate, opacity=0.8):
    mask_index = 1
    mask = masks == mask_index
    mask = mask / mask_index
    mask = np.expand_dims(mask, -1)
    mask_rgb = np.tile(mask, 3)
    mask_rgb = mask_rgb * plate
    image = image * (1.0 - mask) + image * mask * opacity + mask_rgb * (1 - opacity)
    image = np.asarray(image, dtype=np.uint8)
    return image


def save_output(image_name, predict, d_dir):
    predict = np.squeeze(predict)
    predict = np.asarray(predict, dtype=np.uint8)

    img_name = image_name.split("/")[-1]
    image = cv2.imread(image_name)
    predict_mask = cv2.resize(predict, (image.shape[1], image.shape[0]))
    image = draw_mask(image, predict_mask, plate=(122, 233, 32))
    img_name = img_name.replace('.jpg', '.png')
    cv2.imwrite(d_dir+img_name, image)


if __name__ == '__main__':
    # --------- 1. get image path and name ---------

    image_dir = './validation/'
    prediction_dir = './test_results/'
    model_dir = './saved_models/basnet_bsi/basnet_bsi_itr_24000_train_5.146515_tar_0.597883.pth'

    os.makedirs(prediction_dir, exist_ok=True)

    img_name_list = glob.glob(image_dir + '*.jpg')

    # --------- 2. dataloader ---------
    # 1. dataload
    test_salobj_dataset = SalObjDatasetTest(img_name_list=img_name_list, size_wh=(1024, 512))
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)

    # --------- 3. model define ---------
    print("...load BASNet...")
    net = BASNet(3, 1)
    net.load_state_dict(torch.load(model_dir, map_location='cuda'))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:", img_name_list[i_test].split("/")[-1])

        inputs_test = data_test
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7, d8 = net(inputs_test)
        pred = torch.sigmoid(d1) > 0.7
        predict_np = pred.cpu().data.numpy()
        # save results to test_results folder
        save_output(img_name_list[i_test], predict_np, prediction_dir)

        del d1, d2, d3, d4, d5, d6, d7, d8
