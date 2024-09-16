import torchvision
import cv2
import torch
import argparse
import time

from detect_utils import predict, draw_boxes, device
from PIL import Image

parser= argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input video')
parser.add_argument('-m', '--min-size', dest='min_size', default=800, help='minimum input size')
parser.add_argument('-t', '--threshold', default=0.8, help='predicted threshold')
args = vars(parser.parse_args())


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True, min_size=args['min_size'])



cap = cv2.VideoCapture(args['input'])
assert cap.isOpened()== True, 'Error while trying to read video.'

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

save_name= f"{args['input'].split('/')[-1].split('.')[0]}_{args['min_size']}"
out = cv2.VideoWriter(f"outputs/{save_name}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

frame_count = 0    # to keep track of the total number of frames 
total_fps = 0      # total frames per seconds

model = model.eval().to(device)


while(cap.isOpened()):
    ret, frame = cap.read()    
    if ret == True:
        start_time = time.time()
        with torch.no_grad():
            boxes, classes, labels= predict(frame, model, device, args['threshold'])

        image = draw_boxes(frame, boxes, classes, labels)

        end_time = time.time()
        fps = 1/(end_time-start_time)    # how many frames are displayed per second.
        total_fps += fps
        frame_count += 1

        wait_time = max(1, int(fps/4)) #(in milliseconds) the time to wait between displaying each frame. It's used to control the display rate of the frames
        cv2.imshow('image', image)
        out.write(image)
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()

avg_fps= total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")