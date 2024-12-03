import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
import argparse
import torch

from models import model
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim
from coco_dataset import COCODataset, collator, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from train_func import train

## argparse
parser = argparse.ArgumentParser()

## prepare dataset
parser.add_argument("--data_path", type=str, help="your custom dataset path", default="/data/02_COCOData/")

## data generator
parser.add_argument("--num_workers", type=int, help="num workers of generator", default=3)
parser.add_argument("--batch_size", type=int, help="num of batch size", default=8)

## model architecture
parser.add_argument("--backbone", type=str, help="backbone of retinanet, reset18~152", default='resnet34')

## Learning Parameter
parser.add_argument("--epochs", type=int, help="num epochs", default=10)

## Model save
parser.add_argument("--monitor", type=str, help="Criteria of Best model save", default="loss")
parser.add_argument("--model_save_path", type=str, help="your model save path", default="./model_result/ResNet34_Backbone/RetinaNet_ResNet34_model.pth")

## Optimizer parameter
parser.add_argument("--weight_decay", type=float, help="weight decay of Optimizer", default=0.)
parser.add_argument("--lr", type=float, help="learning rate", default=5e-5)
parser.add_argument("--max_norm", type=float, help="max norm of gradient clipping", default=5)

args = parser.parse_args()

## make dataloader
train_dataset = COCODataset(args.data_path, set_name='train2017',
                            transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
val_dataset = COCODataset(args.data_path, set_name='val2017',
                          transform=transforms.Compose([Normalizer(), Resizer()]))

train_sampler = AspectRatioBasedSampler(train_dataset, batch_size=args.batch_size, drop_last=False)
train_dataloader = DataLoader(train_dataset, num_workers=args.num_workers, collate_fn=collator, batch_sampler=train_sampler)

val_sampler = AspectRatioBasedSampler(val_dataset, batch_size=args.batch_size, drop_last=False)
val_dataloader = DataLoader(val_dataset, num_workers=args.num_workers, collate_fn=collator, batch_sampler=val_sampler)

## make model instance
device = "cuda" if torch.cuda.is_available() else "cpu"

if args.backbone == 'resnet18':
    retinanet = model.resnet18(num_classes=len(train_dataset.coco_labels_inverse), pretrained=True).to(device)
elif args.backbone == 'resnet34':
    retinanet = model.resnet34(num_classes=len(train_dataset.coco_labels_inverse), pretrained=True).to(device)
elif args.backbone == 'resnet50':
    retinanet = model.resnet50(num_classes=len(train_dataset.coco_labels_inverse), pretrained=True).to(device)
elif args.backbone == 'resnet101':
    retinanet = model.resnet101(num_classes=len(train_dataset.coco_labels_inverse), pretrained=True).to(device)
elif args.backbone == 'resnet152':
    retinanet = model.resnet152(num_classes=len(train_dataset.coco_labels_inverse), pretrained=True).to(device)
else:
    raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

retinanet.training = True

## get optimizer
optimizer = optim.Adam(retinanet.parameters(), lr=args.lr)

## Train!!
train(args, retinanet, train_dataloader, val_dataloader, optimizer)