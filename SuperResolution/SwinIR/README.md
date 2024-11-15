
## Environment
- [PyTorch >= 1.7](https://pytorch.org/)
- [BasicSR == 1.3.5](https://github.com/XPixelGroup/BasicSR/blob/master/INSTALL.md) 


### Installation
Install Pytorch first.
Then,
```
pip install -r requirements.txt
python setup.py develop
```


## WARNING:
It seems there is a problem in training SwinIR, that it automatically terminates the training in the middle. I don't have any idea how to solve i. So If you want train 500,000 iters, please set 1,000,000 iters for training.


## How To Train
```
 python swinir/train.py -opt options/train/SwinIR/train_SwinIR_SRx2_scratch.yml
```

- The training command is like (If you use distributed cuda)

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 swinir/train.py -opt options/train/SwinFIR/train_SwinIR_SRx2_from_scratch.yml --launcher pytorch
```

The training logs and weights will be saved in the `./experiments` folder.


## How To Inference
**Single Image Super Resolution**
``` 
python inference/inference_swinir.py
```

## How To Test
```
python swinir/test.py -opt options/test/SwinIR/test_swinir.yml
```



