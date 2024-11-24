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
from transform_utils import PILToTensor, ToDtype, Compose
from torch.utils.data import DataLoader

from utils import *
from PIL import Image
from tqdm import tqdm

## argparse
def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def list_of_floats(arg):
    return list(map(float, arg.split(',')))

parser = argparse.ArgumentParser()

## prepare dataset
parser.add_argument("--data_path", type=str, help="your custom dataset path", default="/data/02_COCOData/")

## data generator
parser.add_argument("--num_workers", type=int, help="num workers of generator", default=0)
parser.add_argument("--batch_size", type=int, help="num of batch size", default=8)

## model architecture
parser.add_argument("--backbone", type=str, help="backbone of faster rcnn", default='vgg16')
parser.add_argument('--anchor_sizes', type=list_of_ints, help="anchor size of faster rcnn", default="32, 64, 128, 256, 512")
parser.add_argument('--anchor_ratio', type=list_of_floats, help="anchor aspect ratio of faster rcnn", default="0.5, 1.0, 2.0")
parser.add_argument('--pooler_output_size', type=list_of_ints, help="pooler output size of faster rcnn", default="7, 7")
parser.add_argument("--pooler_sampling_ratio", type=int, help="pooler sampling ratio", default=2)

## Model save
parser.add_argument("--model_save_path", type=str, help="your model save path", default="./model_result/VGG16_backbone/FasterRCNN_VGG_model.pth")

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

# load trained model
weigths = torch.load(args.model_save_path)

if args.backbone == 'vgg16':
    backbone = models.vgg16(weights="VGG16_Weights.IMAGENET1K_V1").features
    backbone.out_channels = 512

    anchor_generator = rpn.AnchorGenerator(sizes=(args.anchor_sizes,), aspect_ratios=(args.anchor_ratio,))
    roi_pooler = ops.MultiScaleRoIAlign(featmap_names=["0"], output_size=args.pooler_output_size, sampling_ratio=args.pooler_sampling_ratio)

    model = FasterRCNN(backbone=backbone, 
                       num_classes=len(val_dataset._get_categories()), 
                       rpn_anchor_generator=anchor_generator, 
                       box_roi_pool=roi_pooler ).to("cuda")
    
    model = model.load_state_dict(weights)
    model.eval()
    
elif args.backbone == 'resnet50fpn':
    backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    model = FasterRCNN(backbone, num_classes=len(val_dataset._get_categories())).to("cuda")
    
    model.load_state_dict(weigths)
    model.eval()

## inference (for one image, example code)
_cate_dict = val_dataset.new_categories
cate_dict = {}
for key, value in _cate_dict.items():
    cate_dict[value] = key

se_idx = 5
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
