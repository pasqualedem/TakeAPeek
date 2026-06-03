# Take a Peek: Efficient Encoder Adaptation for Few-Shot Semantic Segmentation via LoRA

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2512.10521-b31b1b.svg)](https://arxiv.org/abs/2512.10521)
[![Pattern Recognition Letters](https://img.shields.io/badge/Pattern%20Recognition%20Letters-In%20Press-blue)](https://arxiv.org/abs/2512.10521)

**Pasquale De Marinis, Gennaro Vessio, Giovanna Castellano**  
*Department of Computer Science, University of Bari Aldo Moro, Bari, Italy*

</div>

---

> **Paper accepted to *Pattern Recognition Letters* (in press).**  
> Preprint available on [arXiv:2512.10521](https://arxiv.org/abs/2512.10521).

## Overview

![Take a Peek overview](assets/TaP.pdf)

**Take a Peek (TaP)** is a lightweight, model-agnostic method that enhances encoder adaptability for few-shot semantic segmentation (FSS) and cross-domain FSS. Rather than modifying the decoder — as most prior work does — TaP briefly fine-tunes the encoder on the support set at inference time using **Low-Rank Adaptation (LoRA)**, inducing a targeted feature-space shift conditioned on the current episode.

Key properties:
- **Model-agnostic**: plugs into any encoder-decoder FSS pipeline without modifying the decoder.
- **Efficient**: updates only a small fraction of parameters (e.g., 3.08M at rank 2⁶ for DCAMA).
- **Effective**: consistently improves mIoU across COCO 20ⁱ, Pascal 5ⁱ, and cross-domain benchmarks (DeepGlobe, ISIC, Chest X-ray).
- **Catastrophic forgetting-aware**: low-rank updates preserve the encoder's pretrained generalization.

## Getting Started

### Environment

Install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
source .venv/bin/activate
```

### Datasets

#### COCO 20ⁱ

```bash
cd data
mkdir coco && cd coco
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip train2017.zip && unzip val2017.zip && unzip annotations_trainval2014.zip
rm -rf train2017.zip val2017.zip annotations_trainval2014.zip
```

Merge train and val splits:

```bash
mv val2017/* train2017
mv train2017 train_val_2017
rm -rf val2017
```

Rename image filenames in the COCO 2014 annotations:

```bash
python preprocess.py rename_coco20i_json --instances_path data/coco/annotations/instances_train2014.json
python preprocess.py rename_coco20i_json --instances_path data/coco/annotations/instances_val2014.json
```

Expected structure:
```
data/coco/
├── annotations/
│   ├── instances_train2014.json
│   ├── instances_val2014.json
│   └── ...
└── train_val_2017/
```

#### Pascal 5ⁱ

```bash
bash tap/data/script/setup_voc12.sh data/pascal
```

Add SBD augmented data (pre-converted files available [here](https://github.com/DrSleep/tensorflow-deeplab-resnet#evaluation)):

```bash
unzip SegmentationClassAug.zip -d data/pascal
```

Download augmented split lists from [kazuto1011/deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch/files/2945588/list.zip):

```bash
unzip list.zip -d data/pascal/ImageSets/
mv data/pascal/ImageSets/list/* data/pascal/ImageSets/Segmentation/
rm -rf data/pascal/ImageSets/list
```

Rename split files:

```bash
bash tap/data/script/rename.sh data/pascal/ImageSets/Segmentation/train.txt
bash tap/data/script/rename.sh data/pascal/ImageSets/Segmentation/trainval.txt
bash tap/data/script/rename.sh data/pascal/ImageSets/Segmentation/val.txt
```

Expected structure:
```
data/pascal/
├── Annotations/
├── ImageSets/Segmentation/
│   ├── train.txt
│   ├── trainaug.txt
│   ├── trainval.txt
│   ├── trainvalaug.txt
│   └── val.txt
├── JPEGImages/
├── SegmentationClass/
├── SegmentationClassAug/
└── SegmentationObject/
```

#### CD-FSS Datasets (DeepGlobe, ISIC, Chest X-ray)

Refer to [DMTNet](https://github.com/ChenJiayi68/DMTNet) for dataset preparation.

### Pretrained Models

Download pretrained checkpoints from the respective repositories:
[DMTNet](https://github.com/ChenJiayi68/DMTNet) · [HDMNet](https://github.com/Pbihao/HDMNet) · [BAM](https://github.com/chunbolang/BAM) · [Label Anything](https://github.com/pasqualedem/LabelAnything) · [DCAMA](https://github.com/pawn-sxy/DCAMA)

Place them under `checkpoints/`:

```
checkpoints/
├── bam/
├── dcama/
├── hdmnet/
├── la/
└── dmtnet.pt
```

## Running Experiments

All experiment configurations are in the `parameters/` folder. See [`scripts.sh`](scripts.sh) for the full list of commands.

```bash
python main.py --experiment_file=parameters/<filename> --sequential
```

## Results

TaP consistently improves segmentation performance across models and benchmarks. Selected highlights (mean mIoU improvement over the vanilla baseline):

| Model | COCO 20ⁱ 1-way 5-shot | COCO 20ⁱ 2-way 5-shot | Pascal 5ⁱ 2-way 5-shot |
|---|---|---|---|
| BAM | +7.14 | +8.33 | +8.50 |
| DCAMA | +1.74 | +5.44 | +10.30 |
| FPTrans | +0.66 | +3.96 | +2.91 |
| HDMNet | +1.66 | +3.97 | +4.23 |
| Label Anything | +3.32 | +5.00 | +8.34 |

On cross-domain benchmarks with DMTNet (15-shot): **+4.55** on DeepGlobe, **+4.97** on ISIC, **+20.65** on Chest X-ray.

## Citation

If you use this work, please cite:

```bibtex
@article{demarinis2026takeapeek,
  title     = {Take a Peek: Efficient Encoder Adaptation for Few-Shot Semantic Segmentation via LoRA},
  author    = {De Marinis, Pasquale and Castellano, Giovanna and Vessio, Gennaro},
  journal   = {Pattern Recognition Letters},
  year      = {2026},
  note      = {In press},
  url       = {https://arxiv.org/abs/2512.10521}
}
```

## Acknowledgements

This project was granted access to the LEONARDO supercomputer owned by the EuroHPC Joint Undertaking, hosted by CINECA (Italy), through ISCRA.

This repository builds on [DMTNet](https://github.com/ChenJiayi68/DMTNet), [HDMNet](https://github.com/Pbihao/HDMNet), [BAM](https://github.com/chunbolang/BAM), [Label Anything](https://github.com/pasqualedem/LabelAnything), and [DCAMA](https://github.com/pawn-sxy/DCAMA).
