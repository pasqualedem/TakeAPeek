## Prepare the Environment
You need uv package manager. Install the venv using the following command:
```bash
uv sync
source .venv/bin/activate
```

## Prepare the Datasets

Enter the `data` directory, create and enter the directory `coco` and download the COCO 2017 train and val images and the COCO 2014 annotations from the [COCO website](https://cocodataset.org/#download):

```bash
cd data
mkdir coco
cd coco
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
```

Unzip the files:

```bash
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2014.zip
rm -rf train2017.zip val2017.zip annotations_trainval2014.zip
```

The `coco` directory should now contain the following files and directories:

```
coco
├── annotations
│   ├── captions_train2014.json
│   ├── captions_val2014.json
│   ├── instances_train2014.json
│   ├── instances_val2014.json
|   ├── person_keypoints_train2014.json
|   └── person_keypoints_val2014.json
├── train2017
└── val2017
```

Now, join the images of the train and val sets into a single directory:

```bash
mv val2017/* train2017
mv train2017 train_val_2017
rm -rf val2017
```

Finally, you will have to rename image filenames in the COCO 2014 annotations to match the filenames in the `train_val_2017` directory. To do this, run the following script:

```bash
python preprocess.py rename_coco20i_json --instances_path data/coco/annotations/instances_train2014.json
python preprocess.py rename_coco20i_json --instances_path data/coco/annotations/instances_val2014.json
```

Setting up [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/) Dataset with augmented data.

### 1. Instruction to download
``` bash
bash tap/data/script/setup_voc12.sh data/pascal
``` 
```bash
data/
└── pascal/
    ├── Annotations
    ├── ImageSets/
    │   └── Segmentation
    ├── JPEGImages
    ├── SegmentationObject
    └── SegmentationClass
``` 
### 2. Add SBD Augmentated training data
- Convert by yourself ([here](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/data/pascal)).
- Or download pre-converted files ([here](https://github.com/DrSleep/tensorflow-deeplab-resnet#evaluation)), **(Prefer this method)**.

After the download move it into the pascal folder.

```bash
unzip SegmentationClassAug.zip -d data/pascal
```

```bash
data/
└── pascal/
    ├── Annotations
    ├── ImageSets/
    │   └── Segmentation
    ├── JPEGImages
    ├── SegmentationObject
    ├── SegmentationClass
    └── SegmentationClassAug #ADDED
``` 

### 3. Download official sets as ImageSets/SegmentationAug list
From: https://github.com/kazuto1011/deeplab-pytorch/files/2945588/list.zip

```bash
# Unzip the file
unzip list.zip -d data/pascal/ImageSets/
# Move file into Segmentation folder
mv data/pascal/ImageSets/list/* data/pascal/ImageSets/Segmentation/
rm -rf data/pascal/ImageSets/list
```

This is how the dataset should look like
```bash
/data
└── pascal
    ├── Annotations
    ├── ImageSets
    │   └── Segmentation 
    │       ├── test.txt
    │       ├── trainaug.txt # ADDED!!
    │       ├── train.txt
    │       ├── trainvalaug.txt # ADDED!!
    │       ├── trainval.txt
    │       └── val.txt
    ├── JPEGImages
    ├── SegmentationObject
    ├── SegmentationClass
    └── SegmentationClassAug # ADDED!!
        └── 2007_000032.png
```
### 4. Rename
Now run the rename.sh script.
``` bash
bash tap/data/script/rename.sh data/pascal/ImageSets/Segmentation/train.txt
bash tap/data/script/rename.sh data/pascal/ImageSets/Segmentation/trainval.txt
bash tap/data/script/rename.sh data/pascal/ImageSets/Segmentation/val.txt
```

### CD-FSS Datasets
Refer to [DMTNet](https://github.com/ChenJiayi68/DMTNet) for the dataset preparation.


## Prepare the Pretrained Models

Refer to [DMTNet](https://github.com/ChenJiayi68/DMTNet), [HDMNet](https://github.com/Pbihao/HDMNet), [BAM](https://github.com/chunbolang/BAM) and [Label Anything](https://github.com/pasqualedem/LabelAnything), [DCAMA](https://github.com/pawn-sxy/DCAMA) to download their pretrained models.

Structure them into checkpoints folder as follows:
```bash
checkpoints/
├── bam/
├── dcama/
├── hdmnet/
├── la/
└── dmtnet.pt
```

## Run the experiments

You can refer to [scripts](scripts.sh) for the command line arguments. All the experiments configs are contained in the `parameters` folder. You can run the experiments by running the following command:

```bash
python main.py --experiment_file=parameters/<filename> --sequential
```


## Acknowledgements
This repository is built on top of [DMTNet](https://github.com/ChenJiayi68/DMTNet), [HDMNet](https://github.com/Pbihao/HDMNet), [BAM](https://github.com/chunbolang/BAM), [Label Anything](https://github.com/pasqualedem/LabelAnything) and [DCAMA](https://github.com/pawn-sxy/DCAMA).