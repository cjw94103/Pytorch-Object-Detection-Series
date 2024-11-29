import torch
import argparse
import numpy as np
import cv2
from torchvision.models import resnet18, resnet34, resnet50
from models import SSDBackbone
from torchvision.models.detection import ssd
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from coco_dataset import COCODataset
from train_func import train

from transform_utils import PILToTensor, ToDtype, Compose
from torch.utils.data import DataLoader
from torch import optim

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

# make dataloader
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

# load trained model
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

# inference
_cate_dict = val_dataset.new_categories
cate_dict = {}
for key, value in _cate_dict.items():
    cate_dict[value] = key

se_idx = 2
threshold = 0.5
with torch.no_grad():
    image, target, _ = val_dataset[se_idx]
    image = [image.to("cuda")]
    outputs = model(image)

    boxes = outputs[0]["boxes"].to("cpu").numpy()
    labels = outputs[0]["labels"].to("cpu").numpy()
    scores = outputs[0]["scores"].to("cpu").numpy()

    boxes = boxes[scores >= threshold].astype(np.int32)
    labels = labels[scores >= threshold]
    scores = scores[scores >= threshold]

    gtboxes = target["boxes"].numpy()
    gtlabels = target["labels"].numpy()

## gt image
gtboxes = gtboxes.astype(np.int32)
gt_draw_img = np.uint8(np.transpose(image[0].data.cpu().numpy(), (1, 2, 0)) * 255)

for i, bbox in enumerate(gtboxes):
    cv2.rectangle(gt_draw_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
    cv2.putText(gt_draw_img, cate_dict[gtlabels[i]], (bbox[0], bbox[3]), cv2.FONT_ITALIC, 0.3, (255, 255, 0))
Image.fromarray(gt_draw_img)

## prediction image
draw_img = np.uint8(np.transpose(image[0].data.cpu().numpy(), (1, 2, 0)) * 255)
for i, bbox in enumerate(boxes):
    cv2.rectangle(draw_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
    cv2.putText(draw_img, cate_dict[labels[i]], (bbox[0], bbox[3]), cv2.FONT_ITALIC, 0.3, (255, 255, 0))
Image.fromarray(draw_img)