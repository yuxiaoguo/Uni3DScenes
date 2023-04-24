# Data processing of Structured3D for Swin3D

## Sturectured3D Download
Please following the instructions to download the data of Structured3D from [here](https://structured3d-dataset.org/#download). Once the data is downloaded, the files should be organized as follows:
```
-- $DATASET_ROOT
    |-- Structured3D_0.zip
    |-- Structured3D_1.zip
    |-- ...
    |-- Structured3D_perspective_0.zip
    |-- Structured3D_perspective_1.zip
    |-- ...
    |-- Structured3D_bbox.zip
```
The processing script supports reading the data from the zip files directly. It's unnecessary to unzip the files.\
*Our Structured3D is donwloaded about two years ago. If downloaded version has different file organization compared to ours, please raise an issue.*

## Data processing
```bash
python process.py --cfg_path ./configs/Structured3D_points2pth.yaml --data_in $DATASET_ROOT --data_out $OUTPUT_FOLDER
```
Tips:
1. Loading all unzip files into memory may slow and memory consuming. It will take about 10 minutes to load all the data into memory. Then, the script will start to write the data into the output folder.
2. The file organization of the output folder is compatible with ScanNet protocol used in MMDet3D. The output folder should be organized as follows:
```bash
    -- $OUTPUT_FOLDER
        |-- anno_mask
        |-- instance_mask  # should be empty now
        |-- semantic_mask
        |-- points
        |-- desc
```
3. A PyTorch output is also provided. The train/val/test divison follows the official split by the ID of scenes (0-3000 for training; 3000-3249 for validation; 3250-3499 for testing). The output folder should be organized as follows:
```bash
    -- $OUTPUT_FOLDER
        |-- swin3d
            |-- train
                |-- scene_xxxxx_xxxxxx_1cm_seg.pth
            |-- test
                |-- scene_xxxxx_xxxxxx_1cm_seg.pth
            |-- val
                |-- scene_xxxxx_xxxxxx_1cm_seg.pth
```
