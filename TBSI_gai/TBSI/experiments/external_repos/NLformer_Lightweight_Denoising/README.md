# [ECCV2026 Spotlight] NLformer: Lightweight Denoising Transformer
This is the implementation of the paper "Bayesian Self-Attention with Intra-Token Modeling for Lightweight Denoising Transformers", pulished on ECCV 2026.

### Architecture

![](./images/arch.png)

### Main results

#### Real-world denoising
![](./images/table_real.png)
![](./images/visual_real.png)

#### Synthesized AWGN
![](./images/table_gauss.png)
![](./images/visual_gauss.png)

### Reqirements
```
python==3.9.17
pytorch==2.0.1
pytorch-cuda==11.7
einops==0.6.1
fvcore==0.1.5
h5py==3.9.0 
opencv-python==4.11.0.86
scikit-image==0.20.0 
scipy==1.9.1
tqdm==4.65.0
```

### Data Preparation
The data preparation pipeline follows [Restormer](https://github.com/swz30/Restormer.git). Training and test dataset includes:

|                      | Training | Evaluation |
|----------------------|----------|------------|
| Synthetic  |   DIV2K, Flickr2K, BSD, WED        |   Set12, (C)BSD68, Urban100, Kodak, McMaster         |
| Real-world |     SIDD Medium     |  SIDD validation, SIDD benchmark, DND          |

After downloading the dataset above and cropping the training images into patches, please make sure that the dataset is structured as follows.

```
|- dataset
  |- test
    |- BSD68
    |- CBSD68
    |- DND
      |- info.mat
      |- images_srgb
    |- Kodak
    |- MaMaster
    |- Set12
    |- SIDD_val_png
      |- input_crops
      |- target_crops
    |- SIDD
      |- ValidationNoisyBlocksSrgb.mat
      |- ValidationGtBlocksSrgb.mat
      |- BenchmarkNoisyBlocksSrgb.mat
    |- Urban100
      |- img_001.png
      |- ...
  |- train
    |- DFWB_patches
    |- SIDD_Medium
      |- input_crops
      |- target_crops
```

### Training

#### Synthesized AWGN

2-gpu example:
```
CUDA_VISIBLE_DEVICES=0,1 nohup python main_train.py --opt "options/train/gaussian/train_nlformer_gauss_color.json" > train_gauss_color.log&
CUDA_VISIBLE_DEVICES=0,1 nohup python main_train.py --opt "options/train/gaussian/train_nlformer_gauss_gray.json" > train_gauss_gray.log&
```

#### Real-world noise
2-gpu example:
```
CUDA_VISIBLE_DEVICES=0,1 nohup python main_train.py --opt "options/train/real/train_nlformer_real.json" > train_real.log&
```

### Evaluation

#### Synthesized AWGN
```
python test_gaussian.py --opt "options/evaluate/gaussian/eva_nlformer_gauss_c.json"
python test_gaussian.py --opt "options/evaluate/gaussian/eva_nlformer_gauss_g.json"
```

#### Real-world noise

1. SIDD dataset
```
python test_real_sidd.py --opt "options/evaluate/real/eva_nlformer_sidd.json"
```
Then the results will be stored in .mat file under "save_evaluation/sidd/" by default. To evaluate the PSNR and SSIM metric, one can run
```
evaluate_sidd.m
```

2. DND dataset
```
python test_real_dnd.py --opt "options/evaluate/real/eva_nlformer_dnd.json"
```
Then the results will be stored under "save_evaluation/dnd/" by default. To evaluate the PSNR and SSIM metric, please upload the .mat files on the DND official website.

#### Large version of NLformer
Download the pre-trained weights [here](https://drive.google.com/drive/folders/1d76hKji5ocHFYBA29HQuFiuPT4f_uplJ?usp=share_link), and put in "pretrained_models/".
```
python test_gaussian.py --opt "options/evaluate/gaussian/eva_nlformer_large_gauss_c.json"
python test_real_sidd.py --opt "options/evaluate/real/eva_nlformer_large_sidd.json"
```

### Acknowledgements

Parts of this codebase are adapted from [KAIR](https://github.com/cszn/KAIR.git) and [Restormer](https://github.com/swz30/Restormer.git). We thank the original authors for releasing their implementation.

### Citation

If you find this code base helpful, please cite our ECCV paper.

```
@inproceedings{he2026bayesian,
  title={Bayesian Self-Attention with Intra-Token Modeling for Lightweight Denoising Transformers},
  author={He, Runyang and Shen, Zuowei and Ji, Hui},
  booktitle={European Conference on Computer Vision},
  pages={600--619},
  year={2026},
  organization={Springer}
}
```