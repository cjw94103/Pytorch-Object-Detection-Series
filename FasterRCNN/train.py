import torch
import numpy as np
import argparse
from torchvision import models
from torchvision import ops
from torchvision.models.detection import rpn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from coco_dataset import COCODataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim

from tqdm import tqdm
from utils import *
from train_func import train

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

## Learning Parameter
parser.add_argument("--epochs", type=int, help="num epochs", default=20)

## Model save
parser.add_argument("--monitor", type=str, help="Criteria of Best model save", default="loss")
parser.add_argument("--model_save_path", type=str, help="your model save path", default="./model_result/02_Aug_VGG_Backbone/Augment_VGG_model.pth")

## Optimizer parameter
parser.add_argument("--weight_decay", type=float, help="weight decay SGD Optimizer", default=0.0005)
parser.add_argument("--momentum", type=float, help="momentum SGD Optimizer", default=0.9)
parser.add_argument("--lr", type=float, help="learning rate", default=0.001)

args = parser.parse_args()

# dataloader
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
    train_transform = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(dtype=torch.float)
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

# Model
device = "cuda" if torch.cuda.is_available() else "cpu"

if args.backbone == 'vgg16':
    backbone = models.vgg16(weights="VGG16_Weights.IMAGENET1K_V1").features
    backbone.out_channels = 512

    anchor_generator = rpn.AnchorGenerator(sizes=(args.anchor_sizes,), aspect_ratios=(args.anchor_ratio,))
    roi_pooler = ops.MultiScaleRoIAlign(featmap_names=["0"], output_size=args.pooler_output_size, sampling_ratio=args.pooler_sampling_ratio)

    model = FasterRCNN(backbone=backbone, 
                       num_classes=len(train_dataset._get_categories()), 
                       rpn_anchor_generator=anchor_generator, 
                       box_roi_pool=roi_pooler ).to(device)
    
elif args.backbone == 'resnet50fpn':
    backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    model = FasterRCNN(backbone, num_classes=len(train_dataset._get_categories())).to('cuda')

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

# train
train(args, model, train_dataloader, val_dataloader, optimizer)