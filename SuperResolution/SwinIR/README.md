
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


## How To Train
```
 python swinir/train.py -opt options/train/SwinIR/train_SwinIR_SRx2_scratch.yml
```

- The training command is like

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 swinfir/train.py -opt options/train/SwinFIR/train_SwinFIR_SRx2_from_scratch.yml --launcher pytorch
```

The training logs and weights will be saved in the `./experiments` folder.


## How To Inference
**Single Image Super Resolution**
``` 
python inference/inference_swinfir.py
```

## How To Test
```
python swinfir/test.py -opt options/test/SwinIR/test_swinir.yml
```



