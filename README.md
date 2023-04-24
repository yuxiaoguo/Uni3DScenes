# Uni3DScenes
Uni3DScenes maintains a uniform and simplified data processing framework to deal with 3D indoor scene datasets. \
*We pre-release the code for previewing the pretraining data processing of [Swin3D](https://arxiv.org/abs/2304.06906). If you have any problems while using, please free feel to propose [Issues](https://github.com/yuxiaoguo/Uni3DScenes/issues).*

## Install
```bash
git clone https://github.com/yuxiaoguo/Uni3DScenes
cd Uni3DScenes
git submodule update --init --recursive
conda create --name uni3drc python=3.8
conda activate uni3drc  # or `source activate uni3drc` in Linux
pip install -r requirements.txt
```

## Examples
* [Pretraining data precessing for Swin3D](documents/SWIN3D.md)

## Support Datasets
* [Structured3D](https://structured3d-dataset.org/)

## Downstreaming platform support
* [MMDet3D](https://github.com/open-mmlab/mmdetection3d)
* [PyTorch](https://pytorch.org/)
