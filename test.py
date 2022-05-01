
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import argparse

from models import *
from mask import *
import utils

import requests
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from PIL import Image
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Evaluate')
parser.add_argument(
    '--data_dir',
    type=str,
    default='./data',
    help='dataset path')
parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    choices=('cifar10','imagenet'),
    help='dataset')
parser.add_argument(
    '--arch',
    type=str,
    default='vgg_16_bn',
    choices=('resnet_50','vgg_16_bn','resnet_56','resnet_110','densenet_40','googlenet'),
    help='The architecture to prune')
parser.add_argument(
    '--test_model_dir',
    type=str,
    default='./result/tmp/',
    help='The directory where the summaries will be stored.')
parser.add_argument(
    '--eval_batch_size',
    type=int,
    default=1,
    help='Batch size for validation.')
parser.add_argument(
    '--gpu',
    type=str,
    default='0',
    help='Select gpu to use')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
cudnn.benchmark = True
cudnn.enabled = True

# Data
print('==> Preparing data..')

test_tfm = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#Defined dataset
class FoodDataset(Dataset):

    def __init__(self,path,tfm=test_tfm,files = None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files
        print(f"One {path} sample",self.files[0])
        self.transform = tfm
  
    def __len__(self):
        return len(self.files)
  
    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        #im = self.data[idx]
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1 # test has no label
        #print("get item")
        #print(torch.Tensor.size(im))
        return im,label
        
#Test

def test():
    prediction = []
    net.eval()
    with torch.no_grad():
        for data,_ in test_loader:
            test_pred = net(data.to(device))
            print("\n",test_pred)
            test_label = np.argmax(test_pred[0].cpu().data.numpy())
            print(test_label)
            print(cifar10_class[test_label])
            
#Load image from internet

os.makedirs('./img/test/dog',exist_ok=True)
url='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ_5AtKCGDuOxNlVBpepKta1PS7n7QJQa5V5Q&usqp=CAU'
r=requests.get(url)
with open('./img/test/dog/dog1.jpg','wb') as f:
    f.write(r.content)

_dataset_dir = "./img/test"
data_type = "dog"

#Load Model
print('==> Building model..')

net = eval(args.arch)(compress_rate=[0.]*200)
net = net.cuda()
convcfg = net.covcfg
cov_id=len(convcfg)
new_state_dict = OrderedDict()
pruned_checkpoint = torch.load(args.test_model_dir + "/pruned_checkpoint/" + args.arch + "_cov" + str(cov_id) + '.pt',
                               map_location='cuda:0')
tmp_ckpt = pruned_checkpoint['state_dict']
for k, v in tmp_ckpt.items():
    new_state_dict[k.replace('module.', '')] = v
net.load_state_dict(new_state_dict)

#load data
test_set = FoodDataset(os.path.join(_dataset_dir,data_type), tfm=test_tfm)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

cifar10_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

device = 'cuda:0'

test()

        