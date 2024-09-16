import torchvision
import numpy as np
import torch
import argparse
import cv2
from detect_utils import predict, draw_boxes, device

from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input image/video')
parser.add_argument('-m', '--min-size', dest='min_size', default=800, help='minimum input size for the FastRCNN Network')
parser.add_argument('-t', '--threshold', help='predicted threshold', default=0.8)
args = vars(parser.parse_args())


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True, min_size=args['min_size'])



image= Image.open(args['input'])
model.eval().to(device)

boxes, classes, labels = predict(image, model, device, args['threshold'])
image = draw_boxes(image, boxes, classes, labels)
cv2.imshow('Image', image)
save_name = f"{args['input'].split('/')[-1].split('.')[0]}_{args['min_size']}"
cv2.imwrite(f"outputs/{save_name}.jpg", image)
cv2.waitKey(0)
