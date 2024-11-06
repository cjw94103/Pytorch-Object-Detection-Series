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

## load trained model
weights_path = args.model_save_path
weights = torch.load(weights_path)
device = "cuda" if torch.cuda.is_available() else "cpu"

if args.backbone == 'vgg16':
    backbone = models.vgg16(weights="VGG16_Weights.IMAGENET1K_V1").features
    backbone.out_channels = 512

    anchor_generator = rpn.AnchorGenerator(sizes=(args.anchor_sizes,), aspect_ratios=(args.anchor_ratio,))
    roi_pooler = ops.MultiScaleRoIAlign(featmap_names=["0"], output_size=args.pooler_output_size, sampling_ratio=args.pooler_sampling_ratio)

    model = FasterRCNN(backbone=backbone, 
                       num_classes=len(val_dataset._get_categories()), 
                       rpn_anchor_generator=anchor_generator, 
                       box_roi_pool=roi_pooler ).to(device)
    
elif args.backbone == 'resnet50fpn':
    backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    model = FasterRCNN(backbone, num_classes=len(val_dataset._get_categories())).to('cuda')

model = model.load_state_dict(weights)
model = model.to(device)
model.eval()

## Evaluate
gt_lists = []
pred_lists = []
stop_flag = 0

with torch.no_grad():
    model.eval()
    fileidx = 1
    # stop_flag += 1
    for images, targets in tqdm(val_dataloader, total=len(val_dataloader)):
        images = [img.to(device) for img in images]
        outputs = model(images)
        
        # filename (arbitraily)
        filename = str(fileidx).zfill(6) + '.jpg'
        fileidx += 1
        
        # gt 작업
        gt_boxes = targets[0]['boxes'].data.cpu().numpy()
        gt_labels = targets[0]['labels'].data.cpu().numpy()

        for gt_box, gt_label in zip(gt_boxes, gt_labels):
            x, y, w, h = gt_box
            x_min, y_min, x_max, y_max = x, y, x+w, y+h
            gt_list = [x_min, y_min, x_max, y_max, cate_dict[gt_label], filename]
            gt_lists.append(gt_list)

        # inference 작업
        pred_boxes = outputs[0]['boxes'].data.cpu().numpy()
        pred_labels = outputs[0]['labels'].data.cpu().numpy()
        pred_scores = outputs[0]['scores'].data.cpu().numpy()

        for pred_box, pred_label, pred_score in zip(pred_boxes, pred_labels, pred_scores):
            x, y, w, h = pred_box
            x_min, y_min, x_max, y_max = x, y, x+w, y+h
            pred_list = [x_min, y_min, x_max, y_max, cate_dict[pred_label], pred_score, filename]
            pred_lists.append(pred_list)

# make dataframe
gt_lists = np.array(gt_lists)
pred_lists = np.array(pred_lists)

preds_df = pd.DataFrame()
preds_df['xmin'] = np.array(pred_lists[:,0], dtype=np.float32)
preds_df['ymin'] = np.array(pred_lists[:,1], dtype=np.float32)
preds_df['xmax'] = np.array(pred_lists[:,2], dtype=np.float32)
preds_df['ymax'] = np.array(pred_lists[:,3], dtype=np.float32)
preds_df['label'] = pred_lists[:,4]
preds_df['score'] = np.array(pred_lists[:,5], dtype=np.float32)
preds_df['image_name'] = pred_lists[:,6]

labels_df = pd.DataFrame()
labels_df['xmin'] = np.array(gt_lists[:,0], dtype=np.float32)
labels_df['ymin'] = np.array(gt_lists[:,1], dtype=np.float32)
labels_df['xmax'] = np.array(gt_lists[:,2], dtype=np.float32)
labels_df['ymax'] = np.array(gt_lists[:,3], dtype=np.float32)
labels_df['label'] = gt_lists[:,4]
labels_df['image_name'] = gt_lists[:,5]

# extract evaluation score
res = get_coco_from_dfs(preds_df, labels_df, False)