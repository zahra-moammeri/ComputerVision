import config
import torch
import torch.optim as optim

from model import Yolov3
from tqdm import tqdm
from utils import (mean_average_precision, cells_to_bboxes, get_evaluation_bboxes, save_checkpoint, 
                   load_checkpoint, check_class_accuracy, get_loaders, plot_couple_example)
from loss import YoloLoss
import warnings
warnings.filterwarnings('ignore')

torch.backends.cudnn.benchmark = True
device= config.DEVICE

def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    losses = []
    for batch_idx, (x, y) in enumerate(tqdm(train_loader, leave=True)):
        x = x.to(device)
        y0, y1, y2 = (y[0].to(device), y[1].to(device) , y[2].to(device))
        with torch.cuda.amp.autocast():
            out = model(x)
            loss = ( loss_fn(out[0], y0, scaled_anchors[0])
            + loss_fn(out[1], y1, scaled_anchors[1])
            + loss_fn(out[2], y2, scaled_anchors[2]) )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        # mean_loss = sum(losses) / len(losses)
        # loop.set_postfix(loss=mean_loss)



def main():
    model = Yolov3(num_classes= config.NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader, train_eval_loader = get_loaders(train_csv_path=config.DATASET + "/8examples.csv",
                                                                test_csv_path=config.DATASET+'/test.csv')
    # if config.LOAD_MODEL:
    #     load_checkpoint(config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE)

    scaled_anchors = (torch.tensor(config.ANCHORS)*torch.tensor(config.S)
                      .unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)).to(device)
    
    for epoch in range(config.NUM_EPOCHS):
        # plot_couple_example(model, test_loader, 0.6, 0.5, scaled_anchors)
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        if config.SAVE_MODEL:
            save_checkpoint(model, optimizer, filename=config.CHECKPOINT_FILE)

        print(f"Currently epoch {epoch}")
        print("On Train Eval loader:")
        print("On Train Loader:")
        check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)

        if epoch > 0 and epoch % 3 == 0:
            check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
            pred_boxes, true_boxes = get_evaluation_bboxes(test_loader, model, iou_threshold=config.NMS_IOU_THRESH,
                                                           anchors=config.ANCHORS, threshold= config.CONF_THRESHOLD)
            mapval = mean_average_precision(pred_boxes, true_boxes, iou_threshold=config.MAP_IOU_THRESH,
                                            box_format='midpoint', num_classes=config.NUM_CLASSES)
            print(f"MAP: {mapval.item()}")
            model.train()


if __name__ == "__main__":
    main()

