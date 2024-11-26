import torch
import numpy as np
import pandas as pd
import argparse

from torch import optim
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from transform_utils import Compose, RandomHorizontalFlip, PILToTensor, ToDtype, RandomPhotometricDistort
from coco_dataset import COCODataset
from torch.utils.data import DataLoader

from eval_utils.metric import get_inference_metrics_from_df, summarise_inference_metrics
from eval_utils.coco_metric import get_coco_from_dfs

from utils import *
from tqdm import tqdm

## argparse
parser = argparse.ArgumentParser()

## prepare dataset
parser.add_argument("--data_path", type=str, help="your custom dataset path", default="/data/02_COCOData/")

## data generator
parser.add_argument("--num_workers", type=int, help="num workers of generator", default=0)
parser.add_argument("--batch_size", type=int, help="num of batch size", default=4)

## model architecture
parser.add_argument("--backbone", type=str, help="backbone of faster rcnn", default="resnet50fpn")
parser.add_argument("--hidden_layer", type=int, help="feature map reduced dimension", default=256)

## Model save
parser.add_argument("--model_save_path", type=str, help="your model save path", default="./model_result/ResNet50FPN_backbone/MaskRCNN_ResNet50FPN_model.pth")

args = parser.parse_args()

## make dataloader
def collator(batch):
    return tuple(zip(*batch))

transform = Compose(
    [
        PILToTensor(),
        ToDtype(scale=True, dtype=torch.float)
    ]
)
dataset = COCODataset(args.data_path, train=False, transform=transform)
dataloader = DataLoader(
    dataset, batch_size=1, shuffle=False, drop_last=True, collate_fn=collator, num_workers=args.num_workers
)

## load trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = len(dataset.new_categories)

weights_path = args.model_save_path
weights = torch.load(weights_path)

if args.backbone == 'resnet50fpn':
    model = maskrcnn_resnet50_fpn(pretrained_backbone=True) # imagenet pretrained
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_channels=model.roi_heads.box_predictor.cls_score.in_features,
        num_classes=num_classes
    )
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_channels=model.roi_heads.mask_predictor.conv5_mask.in_channels,
        dim_reduced=args.hidden_layer,
        num_classes=num_classes
    )

model.load_state_dict(weights)
model.to(device)
model.eval()

## evaluation
_cate_dict = dataset.new_categories
cate_dict = {}
for key, value in _cate_dict.items():
    cate_dict[value] = key

gt_lists = []
pred_lists = []
stop_flag = 0

with torch.no_grad():
    model.eval()
    fileidx = 1
    for images, targets in tqdm(dataloader, total=len(dataloader)):
        images = [img.to(device) for img in images]
        outputs = model(images)

        # filename (arbitraily)
        filename = str(fileidx).zfill(6) + '.jpg'
        fileidx += 1

        boxes = outputs[0]["boxes"].to("cpu").numpy()
        masks = outputs[0]["masks"].squeeze(1).to("cpu").numpy()
        labels = outputs[0]["labels"].to("cpu").numpy()
        scores = outputs[0]["scores"].to("cpu").numpy()
        # pred_mask_list.append(masks)

        # prediction 작업
        for pred_box, pred_label, pred_score in zip(boxes, labels, scores):
            x, y, w, h = pred_box
            x_min, y_min, x_max, y_max = x, y, x+w, y+h
            pred_list = [x_min, y_min, x_max, y_max, cate_dict[pred_label], pred_score, filename]
            pred_lists.append(pred_list)
    
        # boxes = boxes[scores >= threshold].astype(np.int32)
        # masks = masks[scores >= threshold]
        # labels = labels[scores >= threshold]
        # scores = scores[scores >= threshold]
    
        # # 마스크 처리
        # masks[masks >= threshold] = 1.0
        # masks[masks < threshold] = 0.0
        
        gtboxes = targets[0]["boxes"].numpy()
        gtmasks = targets[0]['masks'].numpy()
        gtlabels = targets[0]["labels"].numpy()
        # gt_mask_list.append(gtmasks)

        # gt 작업
        for gt_box, gt_label in zip(gtboxes, gtlabels):
            x, y, w, h = gt_box
            x_min, y_min, x_max, y_max = x, y, x+w, y+h
            gt_list = [x_min, y_min, x_max, y_max, cate_dict[gt_label], filename]
            gt_lists.append(gt_list)

## make score dataframe
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

## box score
res = get_coco_from_dfs(preds_df, labels_df, False)
print(res)
