import os.path as osp
import swinir.archs
import swinir.data
import swinir.models
import swinir.metrics
from basicsr.test import test_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
