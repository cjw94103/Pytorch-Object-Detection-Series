## 1. Introduction

<p align="center"><img src="https://github.com/user-attachments/assets/ffb90e30-3565-4f21-98eb-bb27fa4d084e" width="80%" height="80%"></p>

RetinaNet은 2017년 "Focal Loss for Dense Object Detection"의 논문에서 제시된 방법입니다. Main idea는 Focal loss이며 이는 cross entropy loss에 class에 따라 변하는 동적인 scaling factor를 추가한 형태를 갖습니다.
이러한 loss function을 통해 학습 시 easy example의 기여도를 자동적으로 down-weight하며, hard example에 대해서 가중치를 높혀 학습을 집중시킬 수 있습니다. Focal loss의 효과를 실험하기 위해 논문에서는 one-stage detector인 RetinaNet을 설계합니다. 해당 네트워크는 ResNet-101-FPN을 backbone network로 가지며 anchor boxes를 적용하여 기존의 two-stage detector에 비해 높은 성능을 보여줍니다. 

### Cross-Entropy Loss
Binary classification에서 사용되는 Cross-Entropy Loss의 formulation은 아래와 같습니다.

<p align="center"><img src="https://github.com/user-attachments/assets/a1367bc4-1975-4ccb-bb77-0b44eb55c12c" width="40%" height="40%"></p>

Cross-Entropy Loss는 모든 sample에 대한 예측 결과를 동등하게 가중치를 둡니다. 이로 인해 어떠한 sample이 쉽게 분류될 수 있음에도 불구하고 작지 않은 loss를 유발하게 됩니다. 많은 수의 easy example의 loss가 더해지면 보기 드문 class를 압도해버려 학습이 제대로 이뤄지지 않습니다. 

### Focal Loss

Focal Loss의 formulation은 아래와 같습니다.

<p align="center"><img src="https://github.com/user-attachments/assets/754b9efb-192b-4d96-804f-17bd5116f275" width="20%" height="20%"></p>

Focal loss는 easy example을 down-weight하여 hard negative sample에 집중하여 학습하는 loss function입니다.
Focal loss의 modulating factor $(1-p_t)^\gamma$와 focusing parameter $\gamma$를 Cross-Entropy Loss에 추가한 형태입니다.

<p align="center"><img src="https://github.com/user-attachments/assets/e8aff601-807f-4627-9d10-3bb13b5dc1fd" width="90%" height="90%"></p>

$\gamma \in [0.5]$에 따른 focal loss의 변화는 위의 그림을 통해 확인할 수 있습니다. 파란색 선은 일반적인 Cross-Entropy Loss를 사용한 것인데 경사가 완만하여 sample간의 loss차이가 거의 나지 않는 것을 확인할 수 있습니다.
반면에 Focal Loss는 $\gamma$에 따라 sample간의 loss 차이를 확인할 수 있고, 이는 well-classified example에게는 작은 loss를 hard-classified example에게는 높은 loss를 부여하여 분류하기 어려운 sample에 대한 가중치를 자동적으로 부여하게 됩니다.

### RetinaNet
논문에서는 Focal loss를 실험하기 위해 RetinaNet이라는 one-stage detector를 설계합니다. RetinaNet은 하나의 backbone network와 각각 classification과 bounding box regression을 수행하는 2개의 subnetwork로 구성되어 있습니다.

## 2. Dataset Preparation
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

## 3. Train
- 아래와 같은 명령어를 실행해주세요. args에 대한 자세한 내용은 train.py를 참고해주세요.
```python
$ python train.py --[args]
```

## 4. Inference
학습이 완료되면 inference.py를 이용하여 추론을 수행할 수 있습니다. args에 대한 자세한 내용은 inference.py를 참고해주세요.
```python
$ python inference.py --[args]
```

## 5. 학습결과
### Quantitative Evaluation
각 모델은 validation dataset에 대하여 AP@IOU 0.50:0.95, AP@IOU 0.50, AP@IOU 0.75을 정량적 평가 메트릭으로 사용합니다.
아래와 같은 명령어로 Quantitative Evaluation을 수행할 수 있습니다. args에 대한 자세한 내용은 evaluation.py를 참고해주세요.
```python
$ python evaluation.py --[args]
```

학습된 모델의 box score는 아래와 같습니다.

|모델|AP@IOU 0.50:0.95|AP@IOU 0.50|AP@IOU 0.75|
|------|---|---|---|
|ResNet34|0.312|0.481|0.332|

### Qualitative Evaluation

<p align="center"><img src="https://github.com/user-attachments/assets/d36b65a8-6dd6-4db5-92ed-ceb98f198a17" width="50%" height="50%"></p>
