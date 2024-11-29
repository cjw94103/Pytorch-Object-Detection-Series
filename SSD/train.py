import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
import argparse
from torchvision.models import resnet18, resnet34, resnet50
from models import SSDBackbone
from torchvision.models.detection import ssd
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from coco_dataset import COCODataset
from train_func import train

from transform_utils import PILToTensor, ToDtype, Compose, RandomHorizontalFlip, RandomPhotometricDistort
from torch.utils.data import DataLoader
from torch import optim

# argparse
parser = argparse.ArgumentParser()

## prepare dataset
parser.add_argument("--data_path", type=str, help="your custom dataset path", default="/data/02_COCOData/")

## data generator
parser.add_argument("--num_workers", type=int, help="num workers of generator", default=0)
parser.add_argument("--batch_size", type=int, help="num of batch size", default=16)
parser.add_argument("--aug_flag", action='store_false', help="Whether to use data augmentation, True recommended")

## model architecture
parser.add_argument("--backbone", type=str, help="backbone of faster rcnn", default='resnet50')

## Learning Parameter
parser.add_argument("--epochs", type=int, help="num epochs", default=10)

## Model save
parser.add_argument("--monitor", type=str, help="Criteria of Best model save", default="loss")
parser.add_argument("--model_save_path", type=str, help="your model save path", default="./model_result/ResNet50_Backbone/SSD_ResNet50_model.pth")

## Optimizer parameter
parser.add_argument("--weight_decay", type=float, help="weight decay of Optimizer", default=0.0005)
parser.add_argument("--lr", type=float, help="learning rate", default=0.001)
parser.add_argument("--max_norm", type=float, help="max norm of gradient clipping", default=5)

args = parser.parse_args()

## make dataloader
def collator(batch):
    return tuple(zip(*batch))

if args.aug_flag==True:
    train_transform = Compose(
    [
        PILToTensor(),
        RandomHorizontalFlip(),
        RandomPhotometricDistort(),
        ToDtype(scale=True, dtype=torch.float)
    ]
    )
    val_transform = Compose(
        [
            PILToTensor(),
            ToDtype(scale=True, dtype=torch.float)
        ]
    )
elif args.aug_flag==False:
    train_transform = Compose(
        [
            PILToTensor(),
            ToDtype(scale=True, dtype=torch.float)
        ]
    )
    val_transform = train_transform

train_dataset = COCODataset(args.data_path, train=True, transform=train_transform)
val_dataset = COCODataset(args.data_path, train=False, transform=val_transform)

train_dataloader = DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collator, num_workers=args.num_workers
)
val_dataloader = DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collator, num_workers=args.num_workers
)

## make model instance
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
    num_classes=len(train_dataset.new_categories)).to(device)

## get optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

## Train!!
train(args, model, train_dataloader, val_dataloader, optimizer)