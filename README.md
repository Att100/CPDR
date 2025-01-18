# CPDR: Towards Highly-Efficient Salient Object Detection via Crossed Post-decoder Refinement

**BMVC 2024**

## Introduction

Many existing salient object detection methods rely on deeper networks with large backbones to achieve higher accuracy, but this comes at the cost of increased computational complexity. Numerous network designs are based on standard UNet and Feature Pyramid Network (FPN) architectures, which have limited capacity for feature extraction and integration. To address this, we propose a lightweight post-decoder refinement module called the Crossed Post-Decoder Refinement (CPDR) to improve feature representation within the FPN or UNet framework. Specifically, we introduce the Attention Down Sample Fusion (ADF), which refines low-level features using channel attention mechanisms guided by high-level attention maps, and the Attention Up Sample Fusion (AUF), which uses spatial attention to guide high-level features with low-level information. Building on these, we propose the Dual Attention Cross Fusion (DACF), which further reduces parameters while maintaining strong performance. Experimental results on five benchmark datasets show that our method achieves state-of-the-art performance.

**Citing CPDR**
If you find CPDR useful in your research, please consider citing our [paper](https://bmva-archive.org.uk/bmvc/2024/papers/Paper_630/paper.pdf).
```
@article{li2025cpdr,
  title={CPDR: Towards Highly-Efficient Salient Object Detection via Crossed Post-decoder Refinement},
  author={Li, Yijie and Wang, Hewei and Katsaggelos, Aggelos},
  journal={arXiv preprint arXiv:2501.06441},
  year={2025}
}
```
This code is only for academic and research purposes.

## Configuration

**PaddlePaddle GPU version**

Please follow the PaddlePaddle official installation instruction [link](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/install/pip/linux-pip_en.html).

**Other dependencies**

```
pip install -r requirements.txt
```

## Dataset

**Download datasets**

* DUTS-TR and DUTS-TE: [Link](http://saliencydetection.net/duts/#org3aad434)
* DUT-OMRON: [Link](http://saliencydetection.net/dut-omron/#org96c3bab)
* HKU-IS: [Link](https://sites.google.com/site/ligb86/hkuis)
* ECSSD: [Link](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)
* PASCAL-S: [Link](http://cbs.ic.gatech.edu/salobj/)

place them under `./dataset`

**Format**

Make sure your `./dataset` folder contains the following:

```
.
├── DUT-OMRON
│   ├── DUT-OMRON-bounding-box
│   ├── DUT-OMRON-eye-fixations
│   ├── Gt
│   └── Imgs
├── DUTS
│   ├── DUTS-TE
│   └── DUTS-TR
├── ECSSD
│   ├── Gt
│   └── Imgs
├── HKU-IS
│   ├── gt
│   ├── imgs
│   ├── testImgSet.mat
│   ├── trainImgSet.mat
│   └── valImgSet.mat
├── PASCAL-S
│   ├── Gt
│   └── Imgs
```

## Usage

**Config name**

- CPDR-L: unet_cpdr_a_efficientnetb0_dice_iou
- CPDR-M: unet_cpdr_a_efficientnetb3_dice_iou
- CPDR-S: fpn_cpdr_b_mobilenetv2lite_dice_iou

**train on DUTS-TR**

example: train CPDR-S

```
python train.py --cfg fpn_cpdr_b_mobilenetv2lite_dice_iou
```

If you are training `CPDR-L` or `CPDR-M` please download the following ImageNet-pretrained checkpoints: `efficientnet-b0`, `efficientnet-b3` from `backbone` folder in [google drive](https://drive.google.com/drive/folders/16HIcvBqY-tj5F9EggPJ0NDrJ42wR0Esj?usp=sharing) and place the two files under `ckpts/pretrained`.

**test on DUTS-TE/HKU-IS/PASCAL-S/ECSSD/DUT-OMRON**

example: test CPDR-S on DUTS-TE

```
python test.py -cfg fpn_cpdr_b_mobilenetv2lite_dice_iou --dataset_name DUTS-TE
```

We provide our trained model for testing and evaluation, you can download them from [google drive](https://drive.google.com/drive/folders/16HIcvBqY-tj5F9EggPJ0NDrJ42wR0Esj?usp=sharing), please place those files directly under `ckpts`.


## Prediction

You can download our generated SOD mask prediction for quantitative or qualitative comparision, from [google drive](https://drive.google.com/file/d/1LbmRRdhdudV3NX3aWrahB6STlLlWniV5/view?usp=sharing).

## References

- Our evaluation is based on [PySODMetrics](https://github.com/lartpang/PySODMetrics)