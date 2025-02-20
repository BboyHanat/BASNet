import os

import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob

from data_loader_custom import SalObjDatasetNew

from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Function
from model import BASNet

import pytorch_ssim
import pytorch_iou

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)


def bce_ssim_loss(pred, target):
    bce_out = bce_loss(pred, target)
    ssim_out = 1 - ssim_loss(pred, target)
    iou_out = iou_loss(pred, target)

    loss = bce_out + ssim_out + iou_out

    return loss


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, labels_v):
    loss0 = bce_ssim_loss(d0, labels_v)
    loss1 = bce_ssim_loss(d1, labels_v)
    loss2 = bce_ssim_loss(d2, labels_v)
    loss3 = bce_ssim_loss(d3, labels_v)
    loss4 = bce_ssim_loss(d4, labels_v)
    loss5 = bce_ssim_loss(d5, labels_v)
    loss6 = bce_ssim_loss(d6, labels_v)
    loss7 = bce_ssim_loss(d7, labels_v)
    # ssim0 = 1 - ssim_loss(d0,labels_v)

    # iou0 = iou_loss(d0,labels_v)
    # loss = torch.pow(torch.mean(torch.abs(labels_v-d0)),2)*(5.0*loss0 + loss1 + loss2 + loss3 + loss4 + loss5) #+ 5.0*lossa
    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7  # + 5.0*lossa
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
        loss0.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.item()))
    # print("BCE: l1:%3f, l2:%3f, l3:%3f, l4:%3f, l5:%3f, la:%3f, all:%3f\n"%(loss1.data[0],loss2.data[0],loss3.data[0],loss4.data[0],loss5.data[0],lossa.data[0],loss.data[0]))

    return loss0, loss


# ------- 2. set the directory of training dataset --------

data_dir = './dataset/'
tra_image_dir = 'images/training/'
tra_label_dir = 'annotations/training/'

image_ext = '.jpg'
label_ext = '.png'

model_dir = "./saved_models/basnet_bsi_0621/"
os.makedirs(model_dir, exist_ok=True)

epoch_num = 100000
batch_size_train = 2
batch_size_val = 1
train_num = 0
val_num = 0

tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)

tra_lbl_name_list = []
for img_path in tra_img_name_list:
    img_name = img_path.split("/")[-1]
    bbb = img_name.replace(image_ext, label_ext)
    tra_lbl_name_list.append(data_dir + tra_label_dir + bbb)

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")

train_num = len(tra_img_name_list)

salobj_dataset = SalObjDatasetNew(tra_img_name_list, tra_lbl_name_list, size_wh=(1024, 512))

salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=3)

# ------- 3. define model --------
# define the net
net = BASNet(3, 1)

# fine-tuning
need_ft_model = './saved_models/basnet_bsi_itr_8000_train_6.803157_tar_0.828863.pth'
net.load_state_dict(torch.load(need_ft_model, map_location='cpu'))
if torch.cuda.is_available():
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
train_step_one_epoch = train_num // batch_size_train
writer = SummaryWriter(comment=f'LR_{0.001}_BS_{batch_size_train}')

for epoch in range(0, epoch_num):
    net.train()
    print("epoch {}".format(epoch))
    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels = data[0], data[1]

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)
        global_step = epoch * train_step_one_epoch + i
        # y zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6, d7 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, labels_v)

        loss.backward()
        # nn.utils.clip_grad_norm(net.parameters(), max_norm=2, norm_type=2)
        optimizer.step()

        # # print statistics
        running_loss += loss.item()
        running_tar_loss += loss2.item()

        writer.add_images('images', inputs, global_step)
        writer.add_images('masks/true', labels, global_step)
        writer.add_images('masks/pred_d0', torch.sigmoid(d0), global_step)
        writer.add_images('masks/pred_d1', torch.sigmoid(d1), global_step)
        writer.add_images('masks/pred_d2', torch.sigmoid(d2), global_step)
        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, d7, loss2, loss

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f" % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val,
            running_tar_loss / ite_num4val))

        if ite_num % 1000 == 0:  # save model every 2000 iterations
            torch.save(net.state_dict(), model_dir + "basnet_bsi_itr_%d_train_%3f_tar_%3f.pth" % (
                ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # resume train
            ite_num4val = 0

print('-------------Congratulations! Training Done!!!-------------')
