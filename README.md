# Filter Pruning with HRank Method
Reference to the paper "HRank: Filter Pruning using High-Rank Feature Map". ([Link](https://arxiv.org/abs/2002.10179)).

The orginal codes can be available at: [https://github.com/lmbxmu/HRank](https://github.com/lmbxmu/HRank).

## Citation
If you find HRank useful in your research, please consider citing:

```
@inproceedings{lin2020hrank,
  title={HRank: Filter Pruning using High-Rank Feature Map},
  author={Lin, Mingbao and Ji, Rongrong and Wang, Yan and Zhang, Yichen and Zhang, Baochang and Tian, Yonghong and Shao, Ling},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={1529--1538},
  year={2020}
}
```

## Running Code

In this code, you can run our models on CIFAR-10 .


### Rank Generation

```shell
python rank_generation.py \
--resume [pre-trained model dir] \
--arch [model arch name] \
--limit [batch numbers] \
--gpu [gpu_id]

```



### Model Training

For the ease of reproducibility. we provide some of the experimental results and the corresponding pruned rate of every layer as belows:
#### Attention! The actual pruning rates are much higher than these presented in the paper since we do not count the next-layer channel removal (For example, if 50 filters are removed in the first layer, then the corresponding 50 channels in the second-layer filters should be removed as well).

##### 1. VGG-16

|  Params      | Flops         | Accuracy |
|--------------|---------------|----------|
| 2.64M(82.1%) | 108.61M(65.3%)| 92.34%   | 

```shell
python main.py \
--job_dir ./result/vgg_16_bn/[folder name] \
--resume [pre-trained model dir] \
--arch vgg_16_bn \
--compress_rate [0.95]+[0.5]*6+[0.9]*4+[0.8]*2 \
--gpu [gpu_id]
```
##### 2. ResNet56

|  Params      | Flops        | Accuracy |
|--------------|--------------|----------|
| 0.49M(42.4%) | 62.72M(50.0%)| 93.17%   | 

```shell
python main.py \
--job_dir ./result/resnet_56/[folder name] \
--resume [pre-trained model dir] \
--arch resnet_56 \
--compress_rate [0.1]+[0.60]*35+[0.0]*2+[0.6]*6+[0.4]*3+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4] \
--gpu [gpu_id]
```
##### 3. ResNet110 
Note that, in the paper, we mistakenly regarded the FLOPs as 148.70M(41.2%). We apologize for it and We will update the arXiv version as soon as possible.

|  Params      | Flops        | Accuracy |
|--------------|--------------|----------|  
| 1.04M(38.7%) |156.90M(37.9%)| 94.23%   | 

```shell
python main.py \
--job_dir ./result/resnet_110/[folder name] \
--resume [pre-trained model dir] \
--arch resnet_110 \
--compress_rate [0.1]+[0.40]*36+[0.40]*36+[0.4]*36 \
--gpu [gpu_id]
```
##### 4. DenseNet40

|  Params      | Flops        | Accuracy |
|--------------|--------------|----------|
| 0.66M(36.5%) |167.41M(40.8%)| 94.24%   | 

```shell
python main.py \
--job_dir ./result/densenet_40/[folder name] \
--resume [pre-trained model dir] \
--arch densenet_40 \
--compress_rate [0.0]+[0.1]*6+[0.7]*6+[0.0]+[0.1]*6+[0.7]*6+[0.0]+[0.1]*6+[0.7]*5+[0.0] \
--gpu [gpu_id]
```
##### 5. GoogLeNet

|  Params      | Flops        | Accuracy |
|--------------|--------------|----------|
| 1.86M(69.8%) |  0.45B(70.4%)| 94.07%   | 

```shell
python main.py \
--job_dir ./result/googlenet/[folder name] \
--resume [pre-trained model dir] \
--arch googlenet \
--compress_rate [0.10]+[0.8]*5+[0.85]+[0.8]*3 \
--gpu [gpu_id]
```

After training, checkpoints and loggers can be found in the `job_dir`. The pruned model will be named `[arch]_cov[i]` for stage i, and therefore the final pruned model is the one with largest `i`.

### Get FLOPS & Params
```shell
python cal_flops_params.py \
--arch resnet_56_convwise \
--compress_rate [0.1]+[0.60]*35+[0.0]*2+[0.6]*6+[0.4]*3+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4]
```

### Evaluate Final Performance
```shell
python evaluate.py \
--dataset [dataset name] \
--data_dir [dataset dir] \
--test_model_dir [job dir of test model] \
--arch [arch name] \
--gpu [gpu id]
```

### Calculate Execution time 
```shell
python calculate_time.py \
--test_model_dir [job dir of test model] \
--arch [arch name] \
--test_batch_size [maximum batch size] \
--gpu [gpu id] \
--iter [iterations]
```

## Other optional arguments
```
optional arguments:
    --data_dir			dataset directory
    				default='./data'
    --dataset			dataset name
    				default: cifar10
    				Optional: cifar10', imagenet
    --lr			initial learning rate
    				default: 0.01
    --lr_decay_step		learning rate decay step
				default: 5,10
    --resume			load the model from the specified checkpoint
    --resume_mask		mask loading directory
    --gpu			Select gpu to use
    				default: 0
    --job_dir			The directory where the summaries will be stored.
    --epochs			The num of epochs to train.
    				default: 30
    --train_batch_size		Batch size for training.
    				default: 128
    --eval_batch_size		Batch size for validation. 
				default: 100
    --start_cov			The num of conv to start prune
    				default: 0
    --compress_rate 		compress rate of each conv
    --arch			The architecture to prune
    				default: vgg_16_bn
				Optional: resnet_50, vgg_16_bn, resnet_56, resnet_110, densenet_40, googlenet
    --test_batch_size		Maximum Batch size for testing function.
    				default: 40
    --iter			Decide the iteration of test function.
    				default: 5
```






## Pre-trained Models 

Additionally, we provide the pre-trained models used in our experiments. 


### CIFAR-10:
 [Vgg-16](https://drive.google.com/open?id=1i3ifLh70y1nb8d4mazNzyC4I27jQcHrE) 
| [ResNet56](https://drive.google.com/open?id=1f1iSGvYFjSKIvzTko4fXFCbS-8dw556T) 
| [ResNet110](https://drive.google.com/open?id=1uENM3S5D_IKvXB26b1BFwMzUpkOoA26m) 
| [DenseNet-40](https://drive.google.com/open?id=12rInJ0YpGwZd_k76jctQwrfzPubsfrZH) 
| [GoogLeNet](https://drive.google.com/open?id=1rYMazSyMbWwkCGCLvofNKwl58W6mmg5c) 

## Pruned Models

Here is the pruned model of above method.([Link](https://drive.google.com/drive/folders/1JfzTE2PqMZ_JhEbnUTpqusD8goLdZCTd?usp=sharing)).

The pruning rates of each model are listed below.

|  Model Name	| Pruning Rate 		| Flops		| Param		| Accuracy |
|---------------|-----------------------|---------------|---------------|----------|
|  ResNet56	| [0.1]+[0.60]*35+[0.0]*2+[0.6]*6+[0.4]*3+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4] | 62.72M	| 0.49M	| 92.65%	|
|  ResNet56_1	| [0.1]+[0.70]*35+[0.3]*2+[0.7]*6+[0.7]*3+[0.3]+[0.7]+[0.3]+[0.7]+[0.3]+[0.7]+[0.3]+[0.7] | 44.51M	| 0.33M	| 91.30%  	| 
|  ResNet110	| [0.1]+[0.40]*36+[0.40]*36+[0.4]*36 		| 156.88M	| 1.06M	| 93.78%	| 
|  ResNet110_1	| [0.1]+[0.60]*36+[0.60]*36+[0.6]*36 		| 105.63M	| 0.70M	| 99.87%	| 
|  VGG16	| [0.6]+[0.4]*6+[0.6]*4+[0.6]*2 		| 150.14M	| 7.75M	| 92.69%	| 
|  VGG16_1	| [0.95]+[0.5]*6+[0.9]*4+[0.8]*2 		| 116.63M	| 4.81M	| 91.83%	| 
|  DenseNet40	| [0.0]+[0.1]*6+[0.7]*6+[0.0]+[0.1]*6+[0.7]*6+[0.0]+[0.1]*6+[0.7]*5+[0.0] 	| 167.41M	| 0.66M	| 93.86%	| 
|  DenseNet40_1	| [0.1]+[0.5]*6+[0.7]*6+[0.5]+[0.5]*6+[0.7]*6+[0.4]+[0.5]*6+[0.7]*5+[0.1] 	| 106.31M	| 0.43M	| 93.16%	| 


