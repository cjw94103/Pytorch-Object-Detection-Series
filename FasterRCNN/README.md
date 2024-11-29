# 1. Introduction
## Overview
<p align="center"><img src="https://github.com/user-attachments/assets/bb4cd547-952c-439f-bbb6-aa8c57a5eaab" width="50%" height="50%"></p>

Faster RCNN은 image feature extractor인 backbone CNN, target object의 클래스 분류, bounding regression을 위한 RPN, ROI Pooling의 3가지로 구성되어 있습니다.
전체 프로세스를 살펴보면 입력 이미지를 backbone CNN에 입력하여 feature map을 얻고 해당 feature map을 RPN의 입력으로 사용하여 candidate bounding box를 만들고 object의 존재 여부에 대한 binary classification을 수행합니다.
candidate bounding box을 feature map에서 추출하여 해당 영역에 대하여 ROI pooling을 수행하고, FC Layer로 전달하여 object가 있다고 판단한 box에 대한 multi-class classification 및 bounding box regression을 수행합니다.
## Backbone CNN
Faster RCNN 논문에서는 VGG와 ZFNet을 image feature extractor로 사용합니다. 여기서는 VGG와 ResNet50FPN backbone을 이용하여 모델을 학습하고 qualitative, quantitative evaluation을 수행합니다.
## Region Proposal Network (RPN)
<p align="center"><img src="https://github.com/user-attachments/assets/d2d8f3f6-6aed-47f0-ad45-bca9cdde8df3" width="70%" height="70%"></p>

RPN은 backbone CNN의 출력인 feature map을 입력으로 받습니다. 위 그림처럼 7x7 size의 feature map이 있다고 가정하면, 3x3의 kernel size로 sliding window을 수행하여 모든 grid cell마다 서로 다른 크기의 $k$개의 Anchor Box를 정의합니다.
논문에서는 Anchor Box를 9개로 정의하지만 Anchor Box와 GT Box의 차이를 Regression으로 prediction하므로 다양한 크기의 predicted Box가 산출됩니다.
구현에서 (C, 7, 7) shape의 feature map을 3x3 convolution에 padding을 1로 설정하여 (256, 7, 7) size로 만듭니다. 
각 sliding window별로 object가 존재하는지, 하지 않는지에 대한 binary classification을 수행하기 위해 1x1 convolution을 이용하여 2(foreground/background) X $k$ = $2k$ channel로 만들고,
Box의 좌표에 대한 regression을 수행하기 위하여 1x1 convolution을 이용하여 4(x, y, w, h) X $k$ = $4k$의 channel로 만들고 각 channel axis에 대하여 binary classification, bounding box regression을 수행합니다.
## ROI Pooling
<p align="center"><img src="https://github.com/user-attachments/assets/46823cd7-deee-4a6a-a76c-1ac5f119815d" width="70%" height="70%"></p>

ROI Pooling의 목적은 size가 다른 region proposal을 FC layer의 입력으로 사용하기 위해 fixed size의 feature로 만들기 위하여 사용됩니다.
Faster RCNN에서 사용한 ROI Pooling은 Fast RCNN에서 사용한 방법과 동일하며 사전 정의된 size의 grid를 이용하여 grid의 bin안에 들어가는 값 사이에 max pooling을 수행하여 fixed size의 feature vector를 만듭니다.

## FPN (Feature Pyramid Network)

<p align="center"><img src="https://github.com/user-attachments/assets/684569f2-29f2-4fa0-91cb-1549ce1cd2d0" width="50%" height="50%"></p>

Faster RCNN 원 논문에서는 FPN (Feature Pyramid Network)을 사용하지 않지만 본 구현에서는 다양한 Backbone에서의 Faster RCNN의 성능을 평가하기 위해 사용합니다.
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
    ├── train2017
    └── val2017
```
COCO 2017 dataset을 다운로드 받으면 annotations 정보는 아래와 같습니다.

- captions_*.json : 이미지를 자연어로 설명하는 caption annotation
- instances_*.json : 이미지의 object의 segmentation 및 bbox annotation
- person_keypoints_*.json : 이미지내의 사람의 pose estimation을 위한 segmentation annotation

이 구현에서는 Object Detection을 위하여 instances_*.json을 사용합니다. 자세한 사항은 coco_dataset.py를 참고해주세요.

# 3. Train
- 아래와 같은 명령어를 실행해주세요. args에 대한 자세한 내용은 train.py를 참고해주세요.
```python
$ python train.py --[args]
```
# 4. Inference
학습이 완료되면 inference.py를 이용하여 추론을 수행할 수 있습니다. args에 대한 자세한 내용은 inference.py를 참고해주세요.
```python
$ python inference.py --[args]
```

# 5. 학습결과
## Quantitative Evaluation
각 모델은 validation dataset에 대하여 AP@IOU 0.50:0.95, AP@IOU 0.50, AP@IOU 0.75을 정량적 평가 메트릭으로 사용합니다.
아래와 같은 명령어로 Quantitative Evaluation을 수행할 수 있습니다. args에 대한 자세한 내용은 evaluation.py를 참고해주세요.
```python
$ python evaluation.py --[args]
```

학습된 모델의 box score는 아래와 같습니다.

|모델|AP@IOU 0.50:0.95|AP@IOU 0.50|AP@IOU 0.75|
|------|---|---|---|
|VGG16|0.339|0.474|0.366|
|ResNet50FPN|0.433|0.562|0.469|

## Qualitative Evaluation

### VGG16 Backbone

<p align="center"><img src="https://github.com/user-attachments/assets/c48737c5-15ae-4c59-8cc3-40b58f031972" width="50%" height="50%"></p>

### ResNet50FPN Backbone

<p align="center"><img src="https://github.com/user-attachments/assets/ce597895-27e0-4f6b-ac94-45b7252f2592" width="50%" height="50%"></p>
