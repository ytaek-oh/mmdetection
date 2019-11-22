# Object Recognition for NIA Sidewalk Dataset 

NIA 인도보행영상 객체인식 데이터셋 학습 및 테스트를 위한 다음의 딥 네트워크 모델 및 코드를 지원합니다.
 - 바운딩박스 객체인식 데이터셋 학습 모델: Faster R-CNN
 - 폴리곤 객체인식 데이터셋 학습 모델: Mask R-CNN 

## License
기재 예정입니다.

## Installation
설치 및 데이터셋 준비는 [INSTALL.md](./docs/INSTALL.md)를 참조하세요.

## Getting Started
인도보행 객체인식 데이터셋 모델 학습 및 테스트 튜토리얼은 [GETTING_STARTED.md](./docs/GETTING_STARTED.md)을 참조하세요. 

## Benchmark
인도보행영상 객체인식 데이터셋으로 학습한 모델의 최종 AP-50 성능과 학습된 모델 다운로드 링크는 [BENCHMARK.md](./docs/BENCHMARK.md)을 참조하세요. 

## Citation

이 프로젝트는 [mmdetection](https://github.com/open-mmlab/mmdetection)을 기반으로 개발되었습니다. 


```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```
