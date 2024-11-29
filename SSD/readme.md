## 1. Introduction
SSD(Single Shot MultiBox Detector)는 이미지 내의 객체를 탐지하기 위해 이미지의 다양한 위치에서 여러 개의 bounding box를 prediction합니다. 이러한 방식을 Multi-Box Detector라고 합니다.
Multi-Box Detector는 이미지에서 사전에 정의된 bounding box(anchor box)를 설정하고, 이 bounding box안에 객체가 존재하는지를 prediction하여 객체를 인식합니다.
SSD는 이미지 내의 객체를 빠르고 정확하게 탐지할 수 있으며 SSD는 객체 탐지와 분류를 동시에 수행하는 대표적인 1-stage detector입니다.
R-CNN 계열은 객체 인식을 위해 region proposal 과정이 필요하고, region proposal과 object recognition 과정이 분리된 구조이므로 2-stage detector입니다.
1-stage detector는 이미지 내에서 객체가 존재할 가능성이 있는 모든 위치를 추론하여 객체의 위치와 클래스를 prediction합니다. 또한 1-stage 알고리즘은 한 번의 순전파로 객체 탐지를 수행하기 때문에 처리 속도가 2-stage detector에 비해 빠릅니다.

### Mutl-scale Feature Map

<p align="center"><img src="https://github.com/user-attachments/assets/41cb525b-4cb0-496e-846a-5a08bcd66c81" width="70%" height="70%"></p>

SSD는 다양한 크기의 객체를 탐지하기 위해 다양한 크기의 feature map을 사용하는 구조입니다. 특징 추출 모델 앞부분에서 추출은 feature map은 작은 객체를 탐지하는데 사용되며, 뒷부분에서 추출한 feature map은 크기가 큰 객체를 탐지합니다.
이렇게 layer별로 추출된 feature map은 각각의 conv layer에 입력되어 객체의 위치와 클래스 정보로 변환됩니다. 이러한 과정을 통해 SSD는 다양한 크기와 종횡비의 객체를 높은 정확도로 탐지할 수 있습니다.
위의 그림을 보면 하나의 feature map은 3차원 텐서 형태가 되는데, 이는 Faster R-CNN에서 사용하는 그리드 방식이 적용됩니다. 입력 이미지를 그리드 형태로 분할하고, 각 그리드 셀은 객체의 존재 여부를 판단하는 작은 네트워크로 해당 그리드 셀
내에 객체가 존자할 가능성이 있는 default box(anchor box)를 prediction합니다.
그리드의 좌표값은 상대적인 위치값으로, 이미지의 각 위치를 고정 크기의 그리드 셀로 분할합니다. 그리드의 벡터값은 기본 박스의 조정값과 클래스 분류 점수로 구성됩니다.
default box는 사전 정의된 다양한 크기와 종횡비를 가지며, 객체가 존재할 가능성이 있는 위치를 대표합니다. default box에 대해 객체의 위치와 클래스를 prediction하고, 해당 그리드 셀 내에서 가장 높은 확률의 객체를 선택하여 최종 탐지 결과를 도출합니다.

### Default Box
SSD는 default box를 사용하는데, Faster R-CNN 모델에서 사용하는 anchor box와 유사한 개념입니다. default box와 anchor box의 주요 차이점은 서로 다른 크기의 feature map에 적용한다는 것인데,
SSD는 38x38, 19x19, 10x10, 5x5, 3x3, 1x1 스케일의 feature map의 각 셀마다 default box를 생성합니다.

default box의 크기는 feature map의 크기에 따라 결정되는데, feature map의 크기가 작을수록 default box를 크게 설정하여 큰 객체를 인식할 수 있습니다. feature map의 크기가 클수록 작은 객체를 인식하기 위해 default box를 작게 설정합니다.

default box의 크기는 입력 이미지의 크기와 feature map의 크기를 고려해 초기 default box의 크기를 설정합니다. 이후 다양한 scale 값을 적용하여 default box의 크기를 조정하며, scale 설정은 아래와 같은 수식으로 계산할 수 있습니다.

<p align="center"><img src="https://github.com/user-attachments/assets/ec744e95-e095-4108-9b35-3838f7d5c2ad" width="70%" height="70%"></p>

$s_{min}, s_{max}$는 default box의 최소 크기와 최대 크기를 의미하며, $m$은 사용할 scale의 개수를 의미합니다. $k$는 1부터 $m$까지의 정수값으로, 각 scale의 index를 의미합니다. 즉, feature map를 추출하는 순서가 됩니다.
$s_{min}, s_{max}, m$은 모델을 학습하기 전 초기값을 할당하며 일반적으로 0.2, 0.9, 6의 값을 사용합니다.

이제 default box의 너비와 높이에 대한 scale을 아래와 같은 수식으로 계산합니다.

<p align="center"><img src="https://github.com/user-attachments/assets/de6fc774-50ed-4c0c-93e7-1a1e57fae551" width="20%" height="20%"></p>

$w_k$는 $k$번째 default box의 너비 스케일을 나타내며, $s_k$는 default box의 크기 스케일을 의미합니다. 마지막으로 종횡비는 아래와 같은 수식으로 계산합니다.

<p align="center"><img src="https://github.com/user-attachments/assets/63cee955-fc1b-4359-8d4b-76c313631d66" width="20%" height="20%"></p>

### Non-Maximum Supression
위의 과정을 거치면 객체 탐지를 위한 많은 후보 Bounding Box들이 탄생하게 됩니다. 이제 GT와 가장 IOU가 높은 bounding box들만 남기기 위해 NMS를 사용합니다.

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

|Backbone|AP@IOU 0.50:0.95|AP@IOU 0.50|AP@IOU 0.75|
|------|---|---|---|
|ResNet18|0.325|0.442|0.343|
|ResNet50|0.182|0.271|0.204|

### Qualitative Evaluation

#### ResNet18 Backbone

<p align="center"><img src="https://github.com/user-attachments/assets/cf2d6e63-458a-4f5f-b178-534ca2d92940" width="50%" height="50%"></p>

#### ResNet50 Backbone

<p align="center"><img src="https://github.com/user-attachments/assets/23cac788-b611-4c96-80a2-cbbe9de3d33f" width="50%" height="50%"></p>
