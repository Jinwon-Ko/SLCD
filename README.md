# [CVPR 2024] Semantic Line Combination Detection

Jinwon Ko, Dongkwon Jin and Chang-Su Kim.

Official code for **"Semantic Line Combination Detection"** in CVPR 2024.
[[arxiv]](http://arxiv.org/abs/2404.18399).

### Dataset
Download the following datasets to ```root/Datasets/```.
```SEL``` and ```SEL_Hard``` datasets are provided in [here](https://github.com/dongkwonjin/Semantic-Line-DRM). ```NKL``` dataset is provided in [here](https://kaizhao.net/nkl). A new dataset called ```CDL``` is available at [here](https://drive.google.com/file/d/1uzzaPaD-lsX10_eKtsPwLqYwkcGhyLKp/view?usp=drive_link).

### Installation
1. Create conda environment:
```
$ conda create -n SLCD python=3.6 anaconda
$ conda activate SLCD
$ conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
$ pip install opencv-python==4.7.0.72
```

2. If you want to get the performance of the paper, download our [model parameters](https://drive.google.com/file/d/1ZjuNuoRl9xCARBW6nhzua95WPQdCnRMx/view?usp=drive_link) to ```root/Modeling/pretrained/``` and [preprocessed data](https://drive.google.com/file/d/1xax-MNFA1cdMhEg23ln4ZLqOd02HytlW/view?usp=drive_link) for SEL, SEL_Hard, NKL(SL5K), and CDL datasets to ```root/Preprocessing/```.
Run with 
```
cd root/Modeling/SLCD/code/
python main.py
```

### Directory structure
    .                           # ROOT
    ├── Modeling                # directory for modeling
    │   ├── Detector
    |   |   ├── code            
    │   ├── SLCD           
    |   |   ├── code            
    │   ├── pretrained          # pretrained model parameters      
    |   |   ├── Detector      
    |   |   |   ├── checkpoint_paper_SEL.pth
    |   |   |   ├── checkpoint_paper_NKL.pth
    |   |   |   ├── checkpoint_paper_CDL.pth
    |   |   ├── SLCD   
    |   |   |   ├── checkpoint_paper_SEL.pth
    |   |   |   ├── checkpoint_paper_NKL.pth
    |   |   |   ├── checkpoint_paper_CDL.pth    
    ├── Preprocessing           # directory for preprocessed data
    │   ├── SEL                 
    |   |   ├── pickle             
    │   ├── SEL_Hard            
    |   |   ├── pickle             
    │   ├── NKL                 
    |   |   ├── pickle             
    │   ├── CDL                 
    |   |   ├── pickle             
    ├── Datasets                # Dataset directory
    │   ├── SEL                 # SEL dataset
    |   |   ├── ICCV2017_JTLEE_gt_pri_lines_for_test
    |   |   ├── ICCV2017_JTLEE_gtlines_all
    |   |   ├── ICCV2017_JTLEE_images
    |   |   ├── Readme.txt
    |   |   ├── test_idx_1716.txt
    |   |   ├── train_idx_1716.txt
    │   ├── SEL_Hard            # SEL_Hard dataset
    |   |   ├── data
    |   |   ├── edge
    |   |   ├── gtimgs
    |   |   ├── images
    |   |   ├── README
    │   ├── NKL                 # NKL dataset
    |   |   ├── Data
    |   |   ├── train.txt
    |   |   ├── val.txt  
    │   ├── CDL                 # CDL dataset
    |   |   ├── train           
    |   |   |   ├── Images
    |   |   |   ├── Labels
    |   |   ├── test            
    |   |   |   ├── Images
    |   |   |   ├── Labels
