import numpy as np

from models.densenet_cifar import densenet_40


# training

#resnet110
'''
# baseline

python3 evaluate.py \
--test_model_dir '/home/r10921097/project/HRank-master/result/resnet_110_baseline/result/' \
--arch resnet_110 \
--gpu 0


# 1 
python3 main.py \
--job_dir ./result/resnet_110/result \
--resume '/home/r10921097/project/HRank-master/pretrained_model/resnet_110.pt'\
--arch resnet_110 \
--compress_rate '[0.1]+[0.60]*36+[0.60]*36+[0.6]*36' \
--gpu 0

python3 evaluate.py \
--test_model_dir '/home/r10921097/project/HRank-master/result/resnet_110/result/' \
--arch resnet_110 \
--gpu 0

# 2 
python3 main.py \
--job_dir './result/resnet_110_1/result' \
--resume '/home/r10921097/project/HRank-master/pretrained_model/resnet_110.pt' \
--arch resnet_110 \
--compress_rate '[0.1]+[0.60]*36+[0.60]*36+[0.6]*36' \
--gpu 0
'''

#resnet56
'''
# 1 
python3 main.py \
--job_dir ./result/resnet_56_baseline/result \
--resume '/home/r10921097/project/HRank-master/pretrained_model/resnet_56.pt'\
--arch resnet_56 \
--compress_rate '[0.1]+[0.60]*35+[0.0]*2+[0.6]*6+[0.4]*3+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4]' \
--gpu 0

# 2 
python3 main.py \
--job_dir './result/resnet_56_1/result' \
--resume '/home/r10921097/project/HRank-master/pretrained_model/resnet_56.pt' \
--arch resnet_56 \
--compress_rate '[0.1]+[0.70]*35+[0.3]*2+[0.7]*6+[0.7]*3+[0.3]+[0.7]+[0.3]+[0.7]+[0.3]+[0.7]+[0.3]+[0.7]' \
--gpu 0

python3 evaluate.py \
--test_model_dir '/home/r10921097/project/HRank-master/result/resnet_56/result/' \
--arch resnet_56 \
--gpu 0
''' 

#VGG16
'''
# 1
python3 main.py \
--job_dir ./result/vgg_16_bn/result \
--resume '/home/r10921097/project/HRank-master/pretrained_model/vgg_16_bn.pt'\
--arch vgg_16_bn \
--compress_rate '[0.95]+[0.5]*6+[0.9]*4+[0.8]*2' \
--gpu 0

# 2 
python main.py \
--job_dir ./result/vgg_16_bn_1/result \
--resume '/home/r10921097/project/HRank-master/pretrained_model/vgg_16_bn.pt' \
--arch vgg_16_bn \
--compress_rate '[0.6]+[0.4]*6+[0.6]*4+[0.6]*2' \
--gpu 0

python3 evaluate.py \
--test_model_dir "/home/eehpc/文件/HRank/result/vgg_16_bn/result/" \
--arch vgg_16_bn \
--gpu 0
'''

#densenet40
'''
# 1
python3 main.py \
--job_dir ./result/densenet_40/result \
--resume '/home/r10921097/project/HRank-master/pretrained_model/densenet_40.pt'\
--arch densenet_40 \
--compress_rate '[0.0]+[0.1]*6+[0.7]*6+[0.0]+[0.1]*6+[0.7]*6+[0.0]+[0.1]*6+[0.7]*5+[0.0]' \
--gpu 0

# 2
python main.py \
--job_dir ./result/densenet_40_1/result \
--resume '/home/r10921097/project/HRank-master/pretrained_model/densenet_40.pt' \
--arch densenet_40 \
--compress_rate '[0.1]+[0.5]*6+[0.7]*6+[0.5]+[0.5]*6+[0.7]*6+[0.4]+[0.5]*6+[0.7]*5+[0.1]' \
--gpu 0

python3 evaluate.py \
--test_model_dir '/home/r10921097/project/HRank-master/result/densenet_40/result/' \
--arch densenet_40 \
--gpu 0
'''

# resnet 50
'''
python3 main.py \
--dataset imagenet \
--data_dir [ImageNet dataset dir] \
--job_dir ./result/resnet_50/result \
--resume '/home/r10921097/project/HRank-master/pretrained_model/resnet_50.pth' \
--arch resnet_50 \
--compress_rate '[0.2]+[0.8]*10+[0.8]*13+[0.55]*19+[0.45]*10' \
--gpu 0
'''


# testing

#resnet110
'''
python3 test.py \
--test_model_dir "/home/eehpc/文件/HRank/result/resnet_110/result/" \
--arch resnet_110 \
--gpu 0

'''

#resnet56
'''
python3 test.py \
--test_model_dir "/home/eehpc/文件/HRank/result/resnet56/result/" \
--arch resnet56 \
--gpu 0

'''

#VGG16
'''
python3 test.py \
--test_model_dir "/home/eehpc/文件/HRank/result/VGG16/result/" \
--arch VGG16 \
--gpu 0

'''

#densenet_40
'''
python3 test.py \
--test_model_dir "/home/eehpc/文件/HRank/result/densenet_40/result/" \
--arch densenet_40 \
--gpu 0

'''

# calculate time

#resnet56
'''
python3 calculate_time.py \
--test_model_dir '/home/eehpc/文件/HRank/result/resnet_56/result/' \
--arch resnet_56 \
--gpu 0 \
--iter 5

python3 evaluate.py \
--test_model_dir '/home/eehpc/文件/HRank/result/resnet56/result/' \
--arch resnet_56 \
--eval_batch_size 15 \
--gpu 0
'''