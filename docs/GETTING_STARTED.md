# GETTING STARTED

이 페이지는 mmdetection toolbox를 사용하여 인도보행 객체인식 데이터셋 학습, 추론, 시각화 등의 튜토리얼을 제공합니다. 
튜토리얼을 진행하기 위해 [mmdetection 설치](./INSTALL.md)가 필요합니다. 

## 학습된 객체인식 모델 테스트하기

  
### Test a dataset

- [x] single GPU testing
- [x] multiple GPU testing
- [x] visualize detection results

학습된 객체인식 모델을 테스트하기 위한 다양한 코드가 제공됩니다. 
사용자는 인도보행영상 데이터셋을 학습한 모델의 성능을 측정해 볼 수 있고,  입력한 이미지에 대한 객체인식 결과를 시각화 할 수 있습니다. 

다음의 명령어는 학습된 모델을 테스트하여 AP (Average Precision)을 계산합니다. 

```shell
# single-gpu testing
python NIA_tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} \
    [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]

# multi-gpu testing
./NIA_tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} \
    [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
```

파라미터 설명:

-   `RESULT_FILE`: 모델의 객체인식 결과를 저장하는 파일의 이름을 지정합니다.  파일의 확장자는 `.pkl` 또는 `.pickle` 입니다.
-   `EVAL_METRICS`: 모델의 객체인식 결과의 성능을 측정할 기준을 지정합니다. 가능한 선택은  `proposal_fast`,  `proposal`,  `bbox`,  `segm`,  `keypoints` 입니다.
-   `--show`: 객체인식 결과 이미지를 새로운 윈도우에 시각화할지 명시합니다. 해당 기능은 single-gpu testing 환경에서 사용 가능합니다.

더 자세한 옵션은 `NIA_tools/test.py`를 참조하세요. 

사용 예시:
학습된 모델의 파라미터는 모두 `checkpoints/` 경로상에 위치한다고 가정합니다.

1. 인도보행영상 바운딩박스 데이터셋 테스트하고 인식 결과를 새로운 윈도우에서 보기
```shell
python NIA_tools/test.py configs/NIA_sidewalk_dataset/faster_rcnn_r50_fpn_1x.py \
    checkpoints/faster_rcnn_r50_fpn_1x.pth \
    --show
``` 

2. 인도보행영상 폴리곤 데이터셋 테스트하고 bbox AP와 segm AP 출력하기 
```shell
python NIA_tools/test.py configs/NIA_sidewalk_dataset/mask_rcnn_r50_fpn_1x.py \
    checkpoints/mask_rcnn_r50_fpn_1x.pth \
    --out results.pkl --eval bbox segm
```

3. 8개의 GPU를 사용하여 인도보행영상 폴리곤 데이터셋 테스트하고, bbox AP와 segm AP 출력하기
```shell
bash ./NIA_tools/dist_test.sh configs/NIA_sidewalk_dataset/mask_rcnn_r50_fpn_1x.py \
    checkpoints/mask_rcnn_r50_fpn_1x.pth \
    8 --out results.pkl --eval bbox segm
```

### 이미지 테스트 API
학습된 모델에서 이미지와 비디오에서 객체 인식을 수행하는 코드입니다. 

```python
from mmdet.apis import init_detector, inference_detector, show_result
import mmcv

config_file = 'configs/faster_rcnn_r50_fpn_1x.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
show_result(img, result, model.CLASSES)
# or save the visualization results to image files
show_result(img, result, model.CLASSES, out_file='result.jpg')

# test a video and show the results
video = mmcv.VideoReader('video.mp4')
for frame in video:
    result = inference_detector(model, frame)
    show_result(frame, result, model.CLASSES, wait_time=1)
```
* 노트북 형식의 객체인식 데모는 [NIA_tools/inference_demo.ipynb](../NIA_tools/inference_demo.ipynb) 파일을 참조하세요. 
* 인도보행영상 객체인식 데이터셋 어노테이션 시각화는 [NIA_tools/plot_annotation_demo.ipynb](../NIA_tools/plot_annotation_demo.ipynb) 파일을 참조하세요.


## 모델 학습하기

mmdetection toolbox는 분산학습 방식(multi-gpu / multi-machine)과 비 분산학습 방식 모두를 지원합니다. 
학습 중 발생하는 모든 로그와 체크포인트 (checkpoint) 파일은 `work_dir` 아래의 경로에 저장됩니다. 

**중요**: 모든 config 파일의 learning rate은 8대의 GPU에 대해 2 images/gpu (배치사이즈=8*2=16) 기준으로 설정되어 있습니다. 학습 환경의 GPU 사용 대수 등에 따라 배치사이즈가 기본 설정과 달라지는 경우 [Linear Scaling Rule](https://arxiv.org/abs/1706.02677) 을 따라서 learning rate을 설정해 주세요. 

* 4 GPUs, 2 imgs/gpu -> learning rate: 0.02*(4*2/16) = 0.01 
* 8 GPUs, 4 imgs/gpu -> learning rate: 0.02*(8*4/16) = 0.04

### GPU로 학습하기
```shell
bash ./NIA_tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

파라미터 설명:
-   `--validate`  (**strongly recommended**): 매 k (기본값: 1)번의 학습 epoch마다 validation set에 대해 성능을 측정합니다.
-   `--work_dir ${WORK_DIR}`: 작업 디렉토리를 새로 지정합니다. 
-   `--resume_from ${CHECKPOINT_FILE}`: 저장된 모델의 학습 파라미터 및 학습 진행 상태를 그대로 불러들인 다음 그 상태에 이어서 계속 학습합니다. 

더 자세한 옵션은 `NIA_tools/train.py`를 참조하세요. 

`--resume_from` 옵션은 체크포인트 형태로 저장된 모델의 파라미터와 optimizer 세팅 및 최종 학습 epoch 정보 등을 모두 불러들여서, 그 상태부터 기존에 진행되던 학습을 이어서 진행합니다. 예기치 못한 원인으로 학습이 중간에 멈춘 경우 사용할 수 있습니다. 반면 `--load_from` 옵션은 저장된 모델의 파라미터만을 현재의 모델에 불러와서 처음부터 학습을 시작합니다. finetuning 등의 목적으로 사용될 수 있습니다. 

### 새로운 모델에서 인도보행영상 객체인식 데이터셋 학습하기

본 github 저장소에서 제공하는 모델 (Faster R-CNN, Mask R-CNN) 이외의 모델로 인도보행영상 객체인식 데이터셋을 학습하고자 할 경우, 해당 모델에 대응되는 config 파일을 데이터셋 특성에 맞게 수정하여 학습에 이용합니다.

Cascade Mask R-CNN 모델을 학습하고자 할 경우

1. `./configs` 경로에서 `casecade_mask_rcnn`으로 시작하는 파일 중 원하는 크기의 모델에 대응되는 `.py` 파일을 찾아서 `./configs/NIA_sidewalk_dataset` 경로로 복사합니다. 
2. `./configs/NIA_sidewalk_dataset` 경로로 복사한 config 파일을 아래와 같이 수정합니다.
* `model` 변수의 `bbox_head` 키와 `mask_head` 아래의 모든 `num_classes` 값을 81에서 29로 수정합니다. 
* `dataset_type` 변수 값을 `SideWalkDataset`으로 설정합니다.
* `data_root` 변수 값을 `./data/sidewalk_dataset`으로 설정합니다. 
* `data` 변수의 `train`, `val`, `test` 각각의 키 아래에 `ann_file` 값과 `img_prefix`의 값을 데이터셋의 경로에 맞게 수정합니다.
ex) train dataset의 경우 기본 설정 예시는 `ann_file=data_root + 'annotations/polygon_train.json'`과 `img_prefix=data_root + 'images_polygon/train/'`입니다.
    

### 학습 데이터셋의 일부만 사용할 때의 설정

인도보행영상 객체인식 데이터셋 학습을 위한 config 파일에서의 `lr_config` 변수 아래의 `steps`  키 값과 `train_epochs` 는 모두 COCO 학습 스케줄링을 그대로 따르도록 결정되었습니다. 
* GPU 8대 사용, 2 imgs per gpu (전체 배치사이즈=16)
* 전체 학습 iteration은 90k이고 iteration이 60k와 80k일 때 lr에 0.1씩 곱합니다.

#### 데이터 설정하기

전체 데이터를 사용할 때와 동일한 방법으로 `NIA_tools/convert_annotation_to_json.py` 파일을 사용하여 데이터셋 위치와 어노테이션 파일을 학습에 사용할 수 있게 변환합니다. 

#### epoch 관련 변수 설정하기

mmdetection은 학습 길이를 iteration이 아니라 epoch으로 설정하기 때문에 주어진 데이터셋의 길이 (ex: 이미지 장수)에 맞는 epoch 값을 설정해야 합니다. 64000장의 이미지를 학습에 이용한다고 할 때  이를 배치사이즈인 16으로 나누면 64000/16=4000 iters per epoch 입니다. 
전체 iteration 수는 90k이고 90k/4000=22.5이므로 `total_epochs` 값은  22 또는 23으로 설정합니다.
같은 방법으로 `lr_config` 변수 아래 lr을 줄이는 epoch 값인 `step` 키 값을 설정합니다. 
(epoch 값을 기반으로 설정하기 때문에 실제 학습에 사용하는 GPU 수와 관계 없이, 위의 계산을 진행하여 epoch 관련 설정을 마칩니다.)

#### 배치 사이즈 및 lr 설정하기

 [Linear Scaling Rule](https://arxiv.org/abs/1706.02677) 에 따라 `data` 변수 아래의 `imgs_per_gpu` 키 값과 `optimizer` 변수 아래의 `lr` 키값을 조절합니다. 



이외 자세한 mmdetection 사용방법은 [mmdetection/GETTING_STARTED](https://github.com/open-mmlab/mmdetection/blob/master/docs/GETTING_STARTED.md)을 참조하세요.

