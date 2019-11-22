# Benchmark

## 학습 환경 구성
아래의 환경에서 인도보행영상 객체인식 데이터셋을 학습하고, 테스트 했습니다. 

### 하드웨어 환경

- 8 NVIDIA Titan Xp GPUs
- Intel Xeon 4210 CPU @ 2.20GHz

### 소프트웨어 환경

- Ubuntu 16.04, CentOS 7.6.1810
- Python 3.6
- PyTorch 1.2
- CUDA 10.1


## 비고

- 객체인식 바운딩박스 데이터셋은 클래스가 29개로 정의된 이미지를 전부 사용하여 Faster R-CNN 모델을 학습시켰고, 폴리곤 데이터셋은 전부 사용하여 Mask R-CNN 모델을 학습했습니다.
- train, validation, test set은 `NIA_tools/convert_annotation_to_json.py` 파일을 이용하여 자동으로 분리했습니다.
- 학습은 distributed 세팅으로, 8대의 GPU에서 진행했습니다.
- 학습 스케줄링은 Detectron에서 COCO 데이터셋을 학습하는 스케줄링을 따릅니다. [GETTING_STARTED.md](./GETTING_STARTED.md)을 참조하세요. 

## 학습된 모델 성능 및 다운로드 링크


|    Backbone     |  Style  | Lr schd |  box AP-50 | mask AP-50 |                                                             Download                                                             |
| :-------------: | :-----: | :-----: | :------: | :-----------------: | :------------: | :----: | :-----: | :------------------------------------------------------------------------------------------------------------------------------: |
|     Faster R-CNN R-50-FPN     |  pytorch |   1x    |  0.791  |  -   |      [model](https://drive.google.com/file/d/1ZWhZL_ZiwQub9ZtoNjNzLXjgLHwbF4Wm/view?usp=sharing)       |
|     Mask R-CNN R-50-FPN     |  pytorch |   1x           |  0.670  |  0.622   |      [model](https://drive.google.com/file/d/1ZXxtM993SibrDhHDhl9cIJ4-EJnn6i6I/view?usp=sharing)       |

