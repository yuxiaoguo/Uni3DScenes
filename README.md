# Uni3DScenes
Uni3DScenes maintains a uniform and simplified data processing framework to deal with 3D indoor scene datasets. \
*We pre-release the code for previewing the pretraining data processing of [Swin3D](https://arxiv.org/abs/2304.06906). If you have any problems while using, please free feel to propose `Issues`.*

## Install
```bash
git clone https://github.com/yuxiaoguo/Uni3DScenes
cd Uni3DScenes
git submodule update --init --recursive
pip install -r requirements.txt
```

## Examples
* [Pretraining data precessing for Swin3D](documents/SWIN3D.md)

## Support Datasets
* [Structured3D](https://structured3d-dataset.org/)

## Downstreaming platform support
* [MMDet3D](https://github.com/open-mmlab/mmdetection3d)
* [PyTorch](https://pytorch.org/)
