
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

from data import imagenet
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
    '--eval_batch_size',
    type=int,
    default=20,
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

def evalution(idx=0,eval_batch_size=20,first_time=0.0):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cudnn.benchmark = True
    cudnn.enabled = True

    # Data
    print('==> Preparing data..')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=eval_batch_size, shuffle=False, num_workers=2)
    print_freq = 3000 // eval_batch_size

    # Model
    print('==> Building model..')
    print("args.arch",args.arch)
    net = None
    net = eval(args.arch)(compress_rate=[0.]*200)
    net = net.cuda()
    #print(net)

    convcfg = net.covcfg

    cov_id=len(convcfg)
    new_state_dict = OrderedDict()
    pruned_checkpoint = torch.load(args.test_model_dir + "/pruned_checkpoint/" + args.arch + "_cov" + str(cov_id) +'.pt',
                                   map_location='cuda:0')
    tmp_ckpt = pruned_checkpoint['state_dict']
    if len(args.gpu) == 1:
        for k, v in tmp_ckpt.items():
            new_state_dict[k.replace('module.', '')] = v
    else:
        for k, v in tmp_ckpt.items():
            new_state_dict['module.' + k.replace('module.', '')] = v

    net.load_state_dict(new_state_dict)

    # calculate the execute time 

    start_cpu = time.process_time()
    start_real = time.time()
    
    #test():
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    net.eval()
    num_iterations = len(testloader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            if batch_idx==0:
                t = time.time()
            outputs = net(inputs)
            if first_time==0.0:
                first_time = time.time()-t
                print("first time = ",time.time()-t,"sec")
            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))
            #if batch_idx%print_freq==0:
            #    print(
            #        '({0}/{1}): '
            #        'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
            #            batch_idx, num_iterations, top1=top1, top5=top5))
        #print("Final Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}".format(top1=top1, top5=top5))
    #end test()
    
    end_cpu = time.process_time()
    end_real = time.time()
    if idx==0:
        time_cpu = end_cpu - start_cpu
        time_real = end_real - start_real
    else:
        time_cpu = end_cpu - start_cpu + first_time
        time_real = end_real - start_real + first_time

    #print("execute cpu time = ",time_cpu , "sec")
    #print("execute real time = ", time_real, "sec")

    return time_cpu,time_real,first_time

time_cpu_rec = np.zeros(args.iter)
time_real_rec = np.zeros(args.iter)

first_time=0.0

'''
time_cpu_rec,time_real_rec,first_time = evalution(0,400,first_time)
print("cpu time = ",time_cpu_rec,"sec")
print("real time = ", time_real_rec,"sec")
'''
time_record_cpu = np.zeros(150)
time_record_real = np.zeros(150)

for batch in range(10,151,1):
    for i in range(args.iter):
        time_cpu_rec[i],time_real_rec[i],first_time = evalution(i,batch,first_time)
    
    print("\n","="*20," batch size = ",batch,"="*20)
    cnt = 0
    for i in range(len(time_cpu_rec)):
        cnt += time_cpu_rec[i]
    cnt = cnt/args.iter
    print("average cpu time = ",cnt, "sec")
    print("average throughput of cpu time = ",10000/cnt,"images/sec")
    time_record_cpu[batch] = cnt

    cnt = 0
    for i in range(len(time_real_rec)):
        cnt += time_real_rec[i]
    cnt = cnt/args.iter
    print("average real time = ",cnt, "sec")
    print("average throughput of real time = ",10000/cnt,"images/sec")
    time_record_real[batch] = cnt
    
    print("\n","="*50)
    
print("time record cpu",time_record_cpu)
print("time record real",time_record_real)

plt.plot(time_record_cpu,'b')
plt.plot(time_record_real,'r')
plt.legend(["cpu time", "real time"], loc ="lower right") 
plt.savefig("time_record")
plt.show()
