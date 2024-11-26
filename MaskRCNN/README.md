# 1. Introduction
## Overview

<p align="center"><img src="https://github.com/user-attachments/assets/e086ecf5-f92e-41fb-aa2c-ef5c70bd9432" width="80%" height="80%"></p>

Faster RCNN은 Backbone CNN에서 얻은 Feature map을 RPN (Region Proposal Network)에 입력하여 RoI (Region of Interest)를 얻고 RoI Pooling을 통해 Fixed size의 Feature map을 얻고 이를 Fully Connected Layer에 통과시켜 Objection classification, BBox regression을 수행합니다. Mask RCNN은 
Segmentation을 위해 Mask Branch가 추가된 구조 입니다.위의 그림과 같이 RoI Pooling을 통해 얻은 Feature map을 Mask branch에 입력하여 Segmentation mask를 얻습니다. Objection Detection에 비해 Segmentation은 Pixel 단위의 Prediction이기 때문에 정교한 Spatial Information을 필요로 하기 때문에
Mask branch는 작은 FCN의 구조를 사용합니다. 또한 RoI Feature를 얻기 위해 RoI Pooling이 아닌 RoI Align을 사용합니다.

<p align="center"><img src="https://github.com/user-attachments/assets/c03938ac-dc3c-4d1c-a0fa-41923a24120a" width="60%" height="60%" align="center"></p>

위 그림은 Mask branch에서 얻은 Mask를 표현합니다. 각 Class에 대한 Binary Mask를 출력하며 해당 픽셀이 해당 Class에 해당하는지 여부를 0과 1로 표시합니다. Mask Branch는 $K^2m$ Size의 feature map을 출력합니다. $m$은 Class의 Feature map size이며 $K$는 Class의 개수입니다.

## RoI Align

<p align="center"><img src="https://github.com/user-attachments/assets/599d849b-38bc-4a15-b7ac-bd2a1f684516" width="60%" height="60%"></p>

Faster RCNN에서의 RoI Pooling는 소수점 Size를 갖는 Feature에 대하여 반올림하여 Pooling을 수행합니다. 하지만 Segmentation에서는 픽셀 단위의 위치 정보를 담아야하기 때문에 소수점를 반올림하는 것이 문제가 될 수 있습니다. 따라서 Mask RCNN에서는 위의 그림과 같이 Bilinear Interpolation을 이용하여 Spatial Information
을 표현하는 RoI Align을 사용합니다.

## FPN (Feature Pyramid Network)

<p align="center"><img src="https://github.com/user-attachments/assets/684569f2-29f2-4fa0-91cb-1549ce1cd2d0" width="50%" height="50%"></p>

FPN은 원본 이미지를 convolutional network에 입력하여 forward pass를 수행하고, 각 stage마다 서로 다른 scale을 가지는 4개의 feature map을 추출합니다.
이 과정을 Bottom-up pathway라고 하며 이후 Top-down pathway를 통해 각 feature map에 1x1 conv 연산을 적용하여 모두 256 channel을 가지도록 rescale하고 upsampling을 수행합니다.
마지막으로 Lateral connections 과정을 통해 pyramid level 바로 아래 있는 feature map과 element-wise addition 연산을 수행합니다. 이를 통해 4개의 서로 다른 resolution의 feature map에 3x3 conv 연산을 적용합니다.
이러한 과정으로 FPN은 이미지에 존재하는 다양한 scale의 object를 더 잘 추출하며, 일반적인 backbone에 비해 detection 성능이 더욱 우수한 특징을 가지고 있습니다.

# 2. Dataset Preparation
데이터셋은 coco2017을 사용합니다. 아래의 명령어를 이용하여 데이터셋을 다운로드 해주세요.
```python
$ wget http://images.cocodataset.org/zips/train2017.zip
$ wget http://images.cocodataset.org/zips/val2017.zip
$ wget http://images.cocodataset.org/zips/test2017.zip

$ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
$ wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
$ wget http://images.cocodataset.org/annotations/image_info_test2017.zip
```
학습을 위한 데이터셋의 구조는 아래와 같습니다.
```python
└── coco2017
    ├── annotations
    ├── train
    └── val
```
COCO 2017 dataset을 다운로드 받으면 annotations 정보는 아래와 같습니다.

- captions_*.json : 이미지를 자연어로 설명하는 caption annotation
- instances_*.json : 이미지의 object의 segmentation 및 bbox annotation
- person_keypoints_*.json : 이미지내의 사람의 pose estimation을 위한 segmentation annotation

이 구현에서는 Object Detection 및 segmentation을 위하여 instances_*.json을 사용합니다. 자세한 사항은 coco_dataset.py를 참고해주세요.

# 3. Train
학습을 위해 아래와 같은 명령어를 사용해주세요. args에 대한 자세한 정보는 train.py 코드를 참고해주세요.
```python
$ python train.py --[args]
```

# 4. Inference
학습이 완료되면 inference.py를 이용하여 추론을 수행할 수 있습니다. args에 대한 자세한 정보는 inference.py 코드를 참고해주세요.
```python
$ python inference.py --[args]
```

# 5. 학습 결과
## Quantitative Evaluation
모델은 Box Score와 Mask Score를 측정합니다. 현재 시점 (2024/11/26)에는 Mask Score를 구현하지 못하였습니다. (추후 구현 예정) 따라서 Box Score를 AP@IOU 0.50:0.95, AP@IOU 0.50, AP@IOU 0.75로 측정합니다.
평가를 위해 아래와 같은 명령어를 사용해주세요. args에 대한 자세한 정보는 evaluation.py 코드를 참고해주세요.
```python
$ python evaluation.py --[args]
```
학습된 모델의 box score는 아래와 같습니다.

|모델|AP@IOU 0.50:0.95|AP@IOU 0.50|AP@IOU 0.75|
|------|---|---|---|
|ResNet50FPN|0.505|0.643|0.552|

## Qualitative Evaluation

<p align="center"><img src="https://github.com/user-attachments/assets/a88fc0e3-aed9-4538-a4d5-1ae23c2333ed" width="80%" height="80%"></p>

<p align="center"><img src="https://github.com/user-attachments/assets/39eef33c-315d-4afa-8214-a74395aa2fb6" width="80%" height="80%"></p>

