## 1. Introduction
<p align="center"><img src="https://github.com/user-attachments/assets/7ccaa306-a774-4d51-abb5-8e6e57a438fb" width="35%" height="35%"></p>

이 저장소는 CV의 여러가지 Task 중에서 Object Detection, Segmentation을 위한 RCNN 시리즈를 구현합니다. 모델의 아키텍처는 논문의 내용을 완전히 반영하진 않지만 주요 개념들을 구현합니다.
딥러닝 프레임워크는 파이토치를 사용합니다.

## 2. Table of Contents
- CNN Based
  - 1-Stage Detector
    - SSD (Single Short Detector)
  - 2-Stage Detector
    - Faster RCNN
    - Mask RCNN
- Transformer Based

## 3. Detection Performace

아래의 결과는 본 저장소에서 실험한 다양한 모델의 bbox detection score를 기록하였습니다.

|Method|Backbone|AP@IOU 0.50:0.95|AP@IOU 0.50|AP@IOU 0.75|
|------|---|---|---|---|
|Faster RCNN|VGG16|0.339|0.474|0.366|
|Faster RCNN|ResNet50FPN|0.433|0.562|0.469|
|Mask RCNN|ResNet50FPN|0.505|0.643|0.552|
|SSD|ResNet18|0.325|0.442|0.343|
|SSD|ResNet50|0.182|0.271|0.204|
|RetinaNet|ResNet34|0.312|0.481|0.332|
