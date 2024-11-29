import torch
import argparse
import numpy as np
import cv2
import pandas as pd
from torchvision.models import resnet18, resnet34, resnet50
from models import SSDBackbone
from torchvision.models.detection import ssd
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from coco_dataset import COCODataset
from tqdm import tqdm

from transform_utils import PILToTensor, ToDtype, Compose
from torch.utils.data import DataLoader
from torch import optim
from eval_utils.metric import get_inference_metrics_from_df, summarise_inference_metrics
from eval_utils.coco_metric import get_coco_from_dfs

# argparse
parser = argparse.ArgumentParser()

## prepare dataset
parser.add_argument("--data_path", type=str, help="your custom dataset path", default="/data/02_COCOData/")

## data generator
parser.add_argument("--num_workers", type=int, help="num workers of generator", default=0)
parser.add_argument("--batch_size", type=int, help="num of batch size", default=64)

## model architecture
parser.add_argument("--backbone", type=str, help="backbone of faster rcnn", default='resnet18')

## Model save
parser.add_argument("--model_save_path", type=str, help="your model save path", default="./model_result/ResNet18_Backbone/SSD_ResNet18_model.pth")

args = parser.parse_args()

## make dataloader
def collator(batch):
    return tuple(zip(*batch))

val_transform = Compose(
    [
        PILToTensor(),
        ToDtype(scale=True, dtype=torch.float)
    ]
)

val_dataset = COCODataset(args.data_path, train=False, transform=val_transform)
val_dataloader = DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=collator, num_workers=args.num_workers
)

## load trained model
weights = torch.load(args.model_save_path)

if args.backbone =="resnet18":
    backbone_out_channels = 512
    backbone_base = resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
elif args.backbone =="resnet34":
    backbone_out_channels = 512
    backbone_base = resnet34(weights="ResNet34_Weights.IMAGENET1K_V1")
elif args.backbone =="resnet50":
    backbone_out_channels = 2048
    backbone_base = resnet50(weights="ResNet50_Weights.IMAGENET1K_V1")


backbone = SSDBackbone(backbone=backbone_base, backbone_out_channels=backbone_out_channels)
anchor_generator = DefaultBoxGenerator(
    aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05, 1.20],
    steps=[8, 16, 32, 64, 100, 300, 512],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ssd.SSD(
    backbone=backbone,
    anchor_generator=anchor_generator,
    size=(512, 512),
    num_classes=len(val_dataset.new_categories)).to(device)

model.load_state_dict(weights)
model.eval()

## evaluation
_cate_dict = val_dataset.new_categories
cate_dict = {}
for key, value in _cate_dict.items():
    cate_dict[value] = key

gt_lists = []
pred_lists = []

with torch.no_grad():
    model.eval()
    fileidx = 1
    for images, targets, _ in tqdm(val_dataloader, total=len(val_dataloader)):
        images = [img.to("cuda") for img in images]
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

## result score
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

res = get_coco_from_dfs(preds_df, labels_df, False)

print(res)