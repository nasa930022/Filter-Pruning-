
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt
import numpy as np

import torchvision
import torchvision.transforms as transforms

import argparse

from models import *
from mask import *
import utils

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
    '--test_batch_size',
    type=int,
    default=40,
    help='Batch size for validation.')
parser.add_argument(
    '--gpu',
    type=str,
    default='0',
    help='Select gpu to use')
parser.add_argument(
    '--iter',
    type=int,
    default=5,
    help='Decide the iteration of test function'
)

args = parser.parse_args()
'''
# Build Model
print('==> Building model..')
print("args.arch",args.arch)
net = None
net = eval(args.arch)(compress_rate=[0.]*200)
net = net.cuda()

# Load pruned model
convcfg = net.covcfg
cov_id=len(convcfg)
new_state_dict = OrderedDict()
pruned_checkpoint = torch.load(args.test_model_dir + "/pruned_checkpoint/" + args.arch + "_cov" + str(cov_id) +'.pt',
                               map_location='cuda:0')
tmp_ckpt = pruned_checkpoint['state_dict']

for k, v in tmp_ckpt.items():
    new_state_dict['module.' + k.replace('module.', '')] = v

net.load_state_dict(new_state_dict)
'''

def evalution(eval_batch_size=20):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cudnn.benchmark = True
    cudnn.enabled = True

    # loading Data for different batch size
    print('==> Preparing data..')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=eval_batch_size, shuffle=False, num_workers=2)
    
    # Build Model
    print('==> Building model..')
    print("args.arch",args.arch)
    net = None
    net = eval(args.arch)(compress_rate=[0.]*200)
    net = net.cuda()

    # Load pruned model
    convcfg = net.covcfg
    cov_id=len(convcfg)
    new_state_dict = OrderedDict()
    pruned_checkpoint = torch.load(args.test_model_dir + "/pruned_checkpoint/" + args.arch + "_cov" + str(cov_id) +'.pt',
                                   map_location='cuda:0')
    tmp_ckpt = pruned_checkpoint['state_dict']

    for k, v in tmp_ckpt.items():
        new_state_dict[k.replace('module.', '')] = v

    net.load_state_dict(new_state_dict)

    # calculate the execute time 
    time_cpu = 0
    time_real = 0
    
    # evalution
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            start_cpu = time.process_time()
            start_real = time.time()
            outputs = net(inputs)
            end_cpu = time.process_time()
            end_real = time.time()
            time_cpu += end_cpu - start_cpu
            time_real += end_real - start_real
            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))
    
    return time_cpu,time_real

time_cpu_iter = np.zeros(args.iter)
time_real_iter = np.zeros(args.iter)
time_record_cpu = np.zeros(150)
time_record_real = np.zeros(150)

for batch in range(10,args.test_batch_size+1,1):
    for i in range(args.iter):
        time_cpu_iter[i],time_real_iter[i] = evalution(batch)
    
    print("\n","="*20," batch size = ",batch,"="*20)
    cnt = 0
    for i in range(len(time_cpu_iter)):
        cnt += time_cpu_iter[i]
    cnt = cnt/args.iter
    time_record_cpu[batch] = cnt
    print("cpu time = ", time_record_cpu[batch])

    cnt = 0
    for i in range(len(time_real_iter)):
        cnt += time_real_iter[i]
    cnt = cnt/args.iter
    time_record_real[batch] = cnt
    print("real time = ", time_record_real[batch])
    print("\n","="*50)

plt.plot(time_record_cpu,'b')
plt.plot(time_record_real,'r')
plt.legend(["cpu time", "real time"], loc ="lower right") 
plt.savefig("time_record")
plt.show()
