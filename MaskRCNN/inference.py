import torch
import numpy as np
import pandas as pd
import argparse

from torch import optim
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from transform_util import Compose, RandomHorizontalFlip, PILToTensor, ToDtype, RandomPhotometricDistort
from coco_dataset import COCODataset
from torch.utils.data import DataLoader

from eval_utils.metric import get_inference_metrics_from_df, summarise_inference_metrics
from eval_utils.coco_metric import get_coco_from_dfs
# from eval_utils.seg_metric import SegmentationMetrics

from utils import *
from tqdm import tqdm

## draw utils
import matplotlib.pyplot as plt

from torchvision.transforms.functional import to_pil_image
from matplotlib import colormaps

def draw_bbox(ax, box, box_color, text, color, mask):
    ax.add_patch(
        plt.Rectangle(
            xy=(box[0], box[1]),
            width=box[2] - box[0],
            height=box[3] - box[1],
            fill=False,
            edgecolor=box_color,
            linewidth=2,
        )
    )
    ax.annotate(
        text=text,
        xy=((box[0] + box[2])//2, (box[1] + box[3])//2),
        color='blue',
        weight="bold",
        fontsize=10,
    )

    mask = np.ma.masked_where(mask == 0, mask)
    # mask_color = {"blue": "Blues", "red" : "Reds"}

    cmap = plt.cm.get_cmap(color)
    norm = plt.Normalize(vmin=0, vmax=1)
    rgba = cmap(norm(mask))
    ax.imshow(rgba, interpolation="nearest", alpha=0.5)

# argparse
parser = argparse.ArgumentParser()

## prepare dataset
parser.add_argument("--data_path", type=str, help="your custom dataset path", default="./data/coco2017/")

## data generator
parser.add_argument("--num_workers", type=int, help="num workers of generator", default=0)
parser.add_argument("--batch_size", type=int, help="num of batch size", default=4)

## model architecture
parser.add_argument("--backbone", type=str, help="backbone of faster rcnn", default="resnet50fpn")
parser.add_argument("--hidden_layer", type=int, help="feature map reduced dimension", default=256)

## Model save
parser.add_argument("--model_save_path", type=str, help="your model save path", default="./model_result/02_Aug_VGG_Backbone/Augment_VGG_model.pth")

args = parser.parse_args()

# Make Datalaoder
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

# load trained model
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

model = model.load_state_dict(weights)
model = model.to(device)
model.eval()

# inference (for one image, example code)
_cate_dict = dataset.new_categories
cate_dict = {}
for key, value in _cate_dict.items():
    cate_dict[value] = key

color_dict = {}
# color_names = list(mcolors.CSS4_COLORS)
color_names = list(colormaps)
for i in range(len(list(cate_dict.keys()))):
    color_dict[i] = color_names[i]

se_idx = -1400
threshold = 0.5

with torch.no_grad():
    image, target = dataset[se_idx]
    image = [image.to(device)]
    outputs = model(image)

    boxes = outputs[0]["boxes"].to("cpu").numpy()
    masks = outputs[0]["masks"].squeeze(1).to("cpu").numpy()
    labels = outputs[0]["labels"].to("cpu").numpy()
    scores = outputs[0]["scores"].to("cpu").numpy()

    boxes = boxes[scores >= threshold].astype(np.int32)
    masks = masks[scores >= threshold]
    labels = labels[scores >= threshold]
    scores = scores[scores >= threshold]

    # 마스크 처리
    masks[masks >= threshold] = 1.0
    masks[masks < threshold] = 0.0
    
    gtboxes = target["boxes"].numpy()
    gtmasks = target['masks'].numpy()
    gtlabels = target["labels"].numpy()

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
plt.imshow(to_pil_image(image[0]))

for box, mask, label in zip(boxes, masks, labels):
    draw_bbox(ax, box, 'red', f"{cate_dict[label]}", color_dict[label], mask)