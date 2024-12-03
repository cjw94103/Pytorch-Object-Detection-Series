import torch
import argparse
import torch
import numpy as np

from models import model
from torchvision import transforms
from torch.utils.data import DataLoader
from coco_dataset import COCODataset, Resizer, Normalizer, AspectRatioBasedSampler, collator
from eval_utils import coco_eval

from tqdm import tqdm

## argparse
parser = argparse.ArgumentParser()

## prepare dataset
parser.add_argument("--data_path", type=str, help="your custom dataset path", default="/data/02_COCOData/")

## data generator
parser.add_argument("--num_workers", type=int, help="num workers of generator", default=3)
parser.add_argument("--batch_size", type=int, help="num of batch size", default=8)

## model architecture
parser.add_argument("--backbone", type=str, help="backbone of retinanet, reset18~152", default='resnet34')

## Model save
parser.add_argument("--model_save_path", type=str, help="your model save path", default="./model_result/ResNet34_Backbone/RetinaNet_ResNet34_model.pth")

args = parser.parse_args()

## make dataloader
val_dataset = COCODataset(args.data_path, set_name='val2017',
                          transform=transforms.Compose([Normalizer(), Resizer()]))

val_sampler = AspectRatioBasedSampler(val_dataset, batch_size=args.batch_size, drop_last=False)
val_dataloader = DataLoader(val_dataset, num_workers=args.num_workers, collate_fn=collator, batch_sampler=val_sampler)

## load trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
weights = torch.load(args.model_save_path)

if args.backbone == 'resnet18':
    retinanet = model.resnet18(num_classes=len(val_dataset.coco_labels_inverse), pretrained=True).to(device)
elif args.backbone == 'resnet34':
    retinanet = model.resnet34(num_classes=len(val_dataset.coco_labels_inverse), pretrained=True).to(device)
elif args.backbone == 'resnet50':
    retinanet = model.resnet50(num_classes=len(val_dataset.coco_labels_inverse), pretrained=True).to(device)
elif args.backbone == 'resnet101':
    retinanet = model.resnet101(num_classes=len(val_dataset.coco_labels_inverse), pretrained=True).to(device)
elif args.backbone == 'resnet152':
    retinanet = model.resnet152(num_classes=len(val_dataset.coco_labels_inverse), pretrained=True).to(device)
else:
    raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

retinanet.load_state_dict(weights)
retinanet.training = False
retinanet.eval()

## evaluation
coco_eval.evaluate_coco(val_dataset, retinanet)