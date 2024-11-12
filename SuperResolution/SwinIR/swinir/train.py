import os.path as osp
import swinir.archs
import swinir.data
import swinir.models
import swinir.metrics
from basicsr.train import train_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
