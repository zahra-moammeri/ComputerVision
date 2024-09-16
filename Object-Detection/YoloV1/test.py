import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from train import transform, train_fn
from model import Yolov1
from dataset import FruitImagesDataset, test_df, test_annots
from loss import YoloLoss
from utils import (
    intersection_over_union, 
    non_max_suppression, 
    mean_average_precision,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)


LOAD_MODEL = True
EPOCH = 1
LEARNING_RATE = 2e-5
BATCH_SIZE = 8 # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL_FILE = "model.pth"
img_show = 4

test_dir = './data/fruits/test'

DEVICE = "cuda" if torch.cuda.is_available else "cpu"



def predictions():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=3).to(device=DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay= WEIGHT_DECAY )
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)


    test_dataset = FruitImagesDataset(transform= transform, df= test_df ,train_dir= test_dir)
    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size= BATCH_SIZE,
        shuffle= True,
        drop_last= False
    )



    for epoch in range (EPOCH):
        model.eval()

        for x, y in test_loader:
            x = x.to(DEVICE)
            for idx in range(img_show):
                bboxes = cellboxes_to_boxes(model(x))
                bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.8, threshold=0.4)
                plot_image(x[idx].permute(1, 2, 0).to('cpu'), bboxes)
            import sys
            sys.exit()


        train_fn(test_loader, model, optimizer, loss_fn)

        pred_boxes , target_boxes = get_bboxes(test_loader, model, iou_threshold=0.5, threshold=0.4)
        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format='midpoint')
        print(f"Test mAP: {mean_avg_prec}")


predictions() 