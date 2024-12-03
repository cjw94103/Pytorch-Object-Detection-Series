import torch
import argparse
import torch
import numpy as np
import cv2
import copy

from models import model
from torchvision import transforms
from torch.utils.data import DataLoader
from coco_dataset import COCODataset, Resizer, Normalizer, AspectRatioBasedSampler, collator
from PIL import Image

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

## inference
cate_dict = val_dataset.labels

se_idx = 1
threshold = 0.35
with torch.no_grad():
    data = val_dataset[se_idx]
    img = data['img'].permute(2, 0, 1).unsqueeze(0).to(device)
    outputs = retinanet(img)

    boxes = outputs[2].to("cpu").numpy()
    labels = outputs[1].to("cpu").numpy()
    scores = outputs[0].to("cpu").numpy()

    boxes = boxes[scores >= threshold].astype(np.int32)
    labels = labels[scores >= threshold]
    scores = scores[scores >= threshold]

    gt = data['annot'].to("cpu").numpy()
    gtboxes = gt[:,:4]
    gtlabels = gt[:,4:].flatten().astype(np.int32)

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

origin_image = np.transpose(img[0].data.cpu().numpy(), (1, 2, 0))
origin_image = np.uint8((mean + origin_image*std) * 255)

## view gt image
gtboxes = gtboxes.astype(np.int32)
gt_draw_img = copy.deepcopy(origin_image)

for i, bbox in enumerate(gtboxes):
    cv2.rectangle(gt_draw_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
    cv2.putText(gt_draw_img, cate_dict[gtlabels[i]], (bbox[0], bbox[3]), cv2.FONT_ITALIC, 0.3, (255, 255, 0))
Image.fromarray(gt_draw_img)

## view prediction image
draw_img = copy.deepcopy(origin_image)
for i, bbox in enumerate(boxes):
    cv2.rectangle(draw_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
    cv2.putText(draw_img, cate_dict[labels[i]], (bbox[0], bbox[3]), cv2.FONT_ITALIC, 0.3, (255, 255, 0))
Image.fromarray(draw_img)