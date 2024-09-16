import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np

from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names),3))  # generating rgb color in shape(n,3) between 0-255

transform = transforms.Compose([
    transforms.ToTensor()
])


def predict(image, model, device, threshold):
    image = transform(image).to(device)
    image = image.unsqueeze(0)
    output = model(image)

    # print the results individually
    # print(f"BOXES: {output[0]['boxes']}")
    # print(f"LABELS: {output[0]['labels']}")
    # print(f"SCORES: {output[0]['scores']}")

    # get the predicted class names
    pred_classes= [coco_names[i] for i in output[0]['labels'].cpu().numpy()]

    # get scores for objects
    pred_scores = output[0]['scores'].detach().cpu().numpy()

    # get bounding boxes
    pred_bboxes= output[0]['boxes'].detach().cpu().numpy()
    boxes = pred_bboxes[pred_scores >= threshold].astype(np.int32)

    return boxes, pred_classes, output[0]['labels']



def draw_boxes(image, boxes, classes, labels):
    image= cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    for idx, box in enumerate(boxes):
        color = COLORS[labels[idx]]
        cv2.rectangle(image,
                      (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])),
                       color, 2 )
        cv2.putText(image, classes[idx], 
                    (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, 
                    lineType= cv2.LINE_AA )
    return image
        

