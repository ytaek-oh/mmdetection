## Installation

### Requirements

설치환경 및 자세한 설명은 [mmdetection INSTALL.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/INSTALL.md)을 참조하세요. 

### Install mmdetection

a. conda 가상환경 생성

```shell
conda create -n sidewalk python=3.7 -y
conda activate sidewalk
```

b. pytorch 설치

```shell
conda install pytorch torchvision -c pytorch
```

c. 저장소 다운로드

```shell
git clone https://github.com/ytaek-oh/mmdetection.git
cd mmdetection
```

d. mmdetection 설치

```shell
python setup.py develop
# or "pip install -v -e ."
```

### Another option: Docker Image

[Dockerfile](../docker/Dockerfile) 로부터 빌드하기

```shell
# build an image with PyTorch 1.1, CUDA 10.0 and CUDNN 7.5
docker build -t mmdetection docker/
```

### Prepare datasets
데이터셋 다운로드: [http://www.aihub.or.kr/content/611](http://www.aihub.or.kr/content/611)

`NIA_tools/convert_annotation_to_json.py` 파일을 통해

 1. 데이터셋 (ex: 바운딩박스, 폴리곤)을 train / validation / split으로 자동으로 나누고,
 2. xml 어노테이션을 학습에 적합한 json 어노테이션 포멧으로 변환하고,
 3. mmdetection 프로젝트 내 data 폴더로 바로가기 링크를 만듭니다.

```shell
python NIA_tools/convert_annotation_to_json.py ${IMAGE_FOLDER} --dataset_type ${DATASET}
```
파라미터 설명:

 - `IMAGE_FOLDER`: 다운로드 받은 데이터셋이 위치한 경로입니다.
 - `DATASET`: 변환할 데이터셋 타입을 지정합니다. bbox와 polygon 중 하나를 입력으로 받습니다. 

`IMAGE_FOLDER` 경로에 다운로드 받은 데이터셋 디렉토리 트리 구조는 아래와 같아야 합니다. (추후 수정)
  
```
${IMAGE_FOLDER}
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   ├── cityscapes
│   │   ├── annotations
│   │   ├── train
│   │   ├── val
│   ├── VOCdevkit
│   │   ├── VOC2007
│   │   ├── VOC2012

```
어노테이션 변환 코드는 데이터셋의 이미지 폴더 일부분에 대해서도 작동합니다. 

변환이 끝난 데이터는 `data/` 에서 확인할 수 있습니다. 

