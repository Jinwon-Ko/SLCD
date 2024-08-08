# [CVPR 2024] Semantic Line Combination Detector
### Jinwon Ko, Dongkwon Jin and Chang-Su Kim.

<img src="https://github.com/Jinwon-Ko/SLCD/blob/main/Figures/Overview.png" alt="overview" width="100%" height="70%" border="10"/>

Official code for **"Semantic Line Combination Detector"** in CVPR 2024.
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

2. If you want to get the performance of the paper, download our [pre-trained model](https://drive.google.com/file/d/1ZjuNuoRl9xCARBW6nhzua95WPQdCnRMx/view?usp=drive_link) to ```root/Modeling/pretrained/``` and [preprocessed data](https://drive.google.com/file/d/1xax-MNFA1cdMhEg23ln4ZLqOd02HytlW/view?usp=drive_link) for SEL, SEL_Hard, NKL(SL5K), and CDL datasets to ```root/Preprocessing/```.

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


### Evaluation
Run with
```
cd root/Modeling/SLCD/code/
python main.py
```

### Train
For training line detector

1. Edit `root/Modeling/Detector/code/config.py`. Please modify `run_mode` to `'train'`. Also, set the dataset you want to train (`dataset_name`).
2. Run with
```
$ cd root/Modeling/Detector/code/
$ python main.py
```

For training SLCD

3. Edit `root/Modeling/SLCD/code/config.py`. Please modify `run_mode` to `'train'`. Also, set the dataset you want to train (`dataset_name`).
4. Run with
```
$ cd root/Modeling/SLCD/code/
$ python main.py
```

### Test
1. If you want to evaluate a model you trained, edit `root/Modeling/SLCD/code/config.py`. Please modify `run_mode` to `'test'`. Also, set the dataset you want to test (`dataset_name`).
2. Run with
```
$ cd root/Modeling/SLCD/code/
$ python main.py
```


### Results
1. Semantic line detection
<img src="https://github.com/Jinwon-Ko/SLCD/blob/main/Figures/Results.png" alt="Semantic line detection" width="100%" height="70%" border="10"/>


2. Road lane detection
<img src="https://github.com/Jinwon-Ko/SLCD/blob/main/Figures/Lane_Detection.png" alt="Road lane detection" width="100%" height="70%" border="10"/>


3. Composition-based image retrieval
<img src="https://github.com/Jinwon-Ko/SLCD/blob/main/Figures/Composition_based_Retrieval.png" alt="Composition-based image retrieval" width="100%" height="70%" border="10"/>


4. Symmetric axis detection
<img src="https://github.com/Jinwon-Ko/SLCD/blob/main/Figures/Symmetric_Axis_Detection.png" alt="Symmetric axis detection" width="80%" height="55%" border="10"/>


5. Vanishing point detection
<img src="https://github.com/Jinwon-Ko/SLCD/blob/main/Figures/Vanishing_Point_Detection.png" alt="Vanishing point detection" width="80%" height="55%" border="10"/>


Semantic feature grouping results
<img src="https://github.com/Jinwon-Ko/SLCD/blob/main/Figures/Semantic_Feature_Grouping_Results.png" alt="Semantic feature grouping" width="80%" height="55%" border="10"/>
