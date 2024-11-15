import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import Yolov1
from dataset import VOCDataset, FruitImagesDataset
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




seed = 123
torch.manual_seed(seed)  # get the same dataset loading

# Hyperparameters etc. 
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 8 # 64 in original paper
WEIGHT_DECAY = 0
EPOCHS = 20
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "model.pth"
train_dir = './data/fruits/train'


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out= model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()   #  resets the gradients of the optimizer to zero
        # ensure that the gradients are reset to zero before the next iteration, 
        # so that the optimizer only uses the gradients from the current iteration to update the parameters.
        loss.backward()         # computes the gradients of the loss with respect to the model's parameters
        optimizer.step()        # the optimizer updates the model's parameters using the gradients

        # Update the progress bar to display the current loss value
        loop.set_postfix(loss= loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")




def main():
    model = Yolov1(split_size=7, num_boxes= 2, num_classes= 3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay= WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=3, mode='max', verbose=True)
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    # train_dataset = VOCDataset("data/8examples.csv", transform= transform, img_dir= IMG_DIR, label_dir= LABEL_DIR)
    # test_dataset = VOCDataset("data/test.csv", transform= transform, img_dir= IMG_DIR, label_dir= LABEL_DIR)

    train_dataset = FruitImagesDataset(transform= transform, train_dir= train_dir)


    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size= BATCH_SIZE,
        num_workers= NUM_WORKERS,
        pin_memory= PIN_MEMORY,
        shuffle= True,
        drop_last= False
    )
 

    for epoch in range(EPOCHS):
        print(f'Running epoch{epoch+1}/{EPOCHS}:')
        # for x, y in train_loader:
        #     x = x.to(DEVICE)
        #     for idx in range(8):
        #         bboxes = cellboxes_to_boxes(model(x))
        #         bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4)
        #         plot_image(x[idx].permute(1, 2, 0).to('cpu'), bboxes)
        #     import sys
        #     sys.exit()

        train_fn(train_loader, model, optimizer, loss_fn)
        
        pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold= 0.4)
        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format='midpoint')
        print(f"Train mAP: {mean_avg_prec}")

        scheduler.step(mean_avg_prec)

        if mean_avg_prec > 0.7:
            checkpoint = {
                "state_dict" : model.state_dict(),
                "optimizer" : optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
            import time 
            time.sleep(10)

        
if __name__ == "__main__":
    main()
