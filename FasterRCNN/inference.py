import torch
import numpy as np
import cv2
import argparse

from torchvision import models
from torchvision import ops
from torchvision.models.detection import rpn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from coco_dataset import COCODataset
from transform_util import PILToTensor, ToDtype
from torch.utils.data import DataLoader

from utils import *
from PIL import Image
from tqdm import tqdm

from eval_utils.metric import get_inference_metrics_from_df, summarise_inference_metrics
from eval_utils.coco_metric import get_coco_from_dfs

import pandas as pd

# Define a custom argument type for a list of integers
def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def list_of_floats(arg):
    return list(map(float, arg.split(',')))

# argparse
parser = argparse.ArgumentParser()

## prepare dataset
parser.add_argument("--data_path", type=str, help="your custom dataset path", default="./data/coco2017/")

## data generator
parser.add_argument("--num_workers", type=int, help="num workers of generator", default=0)
parser.add_argument("--batch_size", type=int, help="num of batch size", default=4)

## model architecture
parser.add_argument("--backbone", type=str, help="backbone of faster rcnn", default='vgg16')
parser.add_argument('--anchor_sizes', type=list_of_ints, help="anchor size of faster rcnn", default="1, 2, 3, 4, 5, 6")
parser.add_argument('--anchor_ratio', type=list_of_floats, help="anchor aspect ratio of faster rcnn", default="0.5, 1, 2")
parser.add_argument('--pooler_output_size', type=list_of_ints, help="pooler output size of faster rcnn", default="7, 7")
parser.add_argument("--pooler_sampling_ratio", type=int, help="pooler sampling ratio", default=2)

## Model save
parser.add_argument("--model_save_path", type=str, help="your model save path", default="./model_result/02_Aug_VGG_Backbone/Augment_VGG_model.pth")

## make dataloader
def collator(batch):
    return tuple(zip(*batch))

val_transform = Compose(
    [
        PILToTensor(),
        ToDtype(scale=True, dtype=torch.float)
    ]
)
val_dataset = COCODataset_Aug(args.data_path, train=False, transform=val_transform)
val_dataloader = DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collator, num_workers=args.num_workers
)

## inference (for one image, example code)
_cate_dict = val_dataset.new_categories
cate_dict = {}
for key, value in _cate_dict.items():
    cate_dict[value] = key

se_idx = 205
threshold = 0.5
with torch.no_grad():
    image, target = val_dataset[se_idx]
    image = [image.to(device)]
    outputs = model(image)

    boxes = outputs[0]["boxes"].to("cpu").numpy()
    labels = outputs[0]["labels"].to("cpu").numpy()
    scores = outputs[0]["scores"].to("cpu").numpy()

    boxes = boxes[scores >= threshold].astype(np.int32)
    labels = labels[scores >= threshold]
    scores = scores[scores >= threshold]

    gtboxes = target["boxes"].numpy()
    gtlabels = target["labels"].numpy()

draw_img = np.uint8(np.transpose(image[0].data.cpu().numpy(), (1, 2, 0)) * 255)
for i, bbox in enumerate(boxes):
    cv2.rectangle(draw_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    cv2.putText(draw_img, cate_dict[labels[i]], (bbox[0], bbox[3]), cv2.FONT_ITALIC, 0.3, (255, 0, 255))
    
cv2.imwrite('./result_image.jpg', draw_img)