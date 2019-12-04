# RepMet
RepMet few-shot detection engine and an Imagenet-LOC detection benchmark 

## Introduction
This manual describes the python code package implementing two modules:
1. Few-shot detection benchmark 
   - Produce episodic data
   - Manage a detection engine

2. RepMet detection engine
   - Train a model for episode data (novel categories) based on pretrained model
   - Perform detection and perfromance evaluation

The RepMet algorithm is described in the paper [1]. 
	
For any questions/issues, please open an issue in this repository or email us:  
*Joseph Shtok, josephs@il.ibm.com*  
*Leonid Karlinsky, leonidka@il.ibm.com*
	
	
## Setup

The codebase was developed in a conda environment with Python 2.7, MXNet 1.0.0, and CUDA 8.0. 
To build the environment, follow the steps:
1.	`conda create -n env-python2.7-mxnet1.0.0 python=2.7`
2.	`. activate env-python2.7-mxnet1.0.0`
3.	Put the file requirements.txt under Anaconda*/.../envs.
4.	`pip install -r requirements.txt`
5.	Installing additional packages:
a.	`conda install matplotlib`
b.	`pip install sklearn`
c.	`pip install PyYAML`
d.	`pip install opencv-python`
6.	`conda activate env-python2.7-mxnet1.0.0`

The file `requirements.txt` is found in the root of the repository.

The additional files required for operating the repository are available at `https://drive.google.com/drive/folders/1MZ6HWQpR_Oseo5_v5gmrlAyubrPL-ciO?usp=sharing`. The folders provided in 
this link ('data' and 'output') should be placed  under the RepMet root of the git package.
The 'data' folder contains the pre-trained model and associated files, and the /output/benchmarks contains the the benchmark files 

The dataset information (images, Ground Truth boxes and classes) is given in a roidb_*.pkl file, produced during the training of the base model. The structure of roidb is described below. In order to use the package with other datasets, a code for creating one may be based on `RepMet/lib/dataset/imagenet.py`

Before the benchmark can be executed, all the image paths in the roidb structure need to be replaced; all the rest of the paths are in the format `./data/…`, they should work once the Box content is copied under the repository root `RepMet/`. 

*Code package root:* 
The main execution script is `few_shot_benchmark.py`. It contains pathes for all required files relative to the `root` folder, where the source soude is deployed. 
The `root` string is hardcoded in line 39 of the `few_shot_benchmark.py`; please update it to your root folder before starting the work.

## Downloading Imagenet-LOC dataset
The Imagenet-LOC dataset can be downloaded from `http://image-net.org/challenges/LSVRC/2017/download-images-1p39.php`, 
Specifically, to download and untar the dataset from command line, run  

 `wget -c   http://image-net.org/image/ILSVRC2017/ILSVRC2017_CLS-LOC.tar.gz`
 
 
### Replacement of image pathes:
The list entries from voc_inloc_gt_roidb.pkl have a field ‘image’ (see section ‘Structure of roidb for rull information). The replacement needs to be performed as follows:

| Current value 	| New value |
| --- | --- |
| roidb[i][‘image’] = <old_path>/<image_name.jpg> | roidb[i][‘image’] = <new_path>/<image_name.jpg> |

where the new path is your Imagenet-LOC dataset location.

## Example executions of a benchmark experiment
As a first step, you may run the short toy benchmark for 1-shot, 3-way detection:
from the main folder of the repository, `/RepMet`, execute 

`python fpn/few_shot_benchmark.py --test_name=RepMet_inloc --Nshot=2 --Nway=3 --Nquery_cat=2 --Nepisodes=2 --display=1`

To reconstruct the 1-shot, 5-way experiment with the RepMet detector (no fine-tuning) from the CVPR paper, run

`python fpn/few_shot_benchmark.py --test_name=RepMet_inloc  --Nshot=1 --Nway=5 --Nquery_cat=10 --Nepisodes=500`

Run the same setup with model fine-tuning on each episode: 

`python fpn/few_shot_benchmark.py --test_name=RepMet_inloc  --Nshot=1 --Nway=5 --Nquery_cat=10 --Nepisodes=500 --do_finetune=1 --num_finetune_epochs=5 --lr=5e-4`

The `few_shot_benchmark.py` is the main script executing all operations. Main argument, determining the detector and dataset to use,
 is the `--test_name`. In the example above, `--test_name=RepMet_inloc` evokes the RepMet detector, with the Imagenet-LOC dataset.
 Using `--test_name=Vanilla_inloc` will call the baseline detector (see the paper for details):  
`python fpn/few_shot_benchmark.py --test_name=Vanilla_inloc  --Nshot=1 --Nway=5 --Nquery_cat=10 --Nepisodes=500`

The output is produced in `RepMet/output/benchmarks/<test_name>`. In this location, a folder, corresponding to specific 
test arguments is created (e.g., `RepMet_inloc_1shot_5way_10qpc_ft:5` is a folder for 1-shot, 5-way, 10 query examples per-class, with 5 epochs of fine-tuning).
In this test folder, a log file is produced for each code execution (time stamped). There is a subfolder for each episode, where the graphical visualizations 
of the trainng images and detections in test images will be produced if the --display=1 is set. 



## Episodic tests
The few-shot test consists of a number of isolated episodes (tasks), in which a new set of classes is presented to the detector for training (on few-shot data) and detection. 
The episodes (see *Structure of episodic data* below) are producing by randomly drawing from the database images for training and test (in practice, image IDs are selected and stored.)
A benchmark is determined by values of four arguments. for example,
`--Nshot=1 --Nway=5 --Nquery_cat=10 --Nepisodes=500'
Here Nshot is number of samples per category, Nway is the number of few-shot categories, Nquery_cat is number of query (test) images per category, and Nepisodes is the number of episodes.
By default, the algorithm loads an existing file with episodic test data (if available). If the set of episodes for the specified configuration, was not previously created, or if the argument --gen_episodes=1 is provided, the episodes file will be created (but no tests will run at this time).

For example, to create a new benchmark with 3-shot, 4-way, 2 test samples per class, 2 episodes, run    
`python fpn/few_shot_benchmark.py --test_name=RepMet_inloc --gen_episodes=1, --load_episodes=0, --Nshot=3 --Nway=4 --Nquery_cat=2 --Nepisodes=2`

to create the benchmark and then run    
`python fpn/few_shot_benchmark.py --test_name=RepMet_inloc --Nshot=3 --Nway=4 --Nquery_cat=2 --Nepisodes=2`

to test it. Note that a separate benchmark file is produced for each test_case.

## Full model training
The code for pretraining the model on a large dataset of auxiliary categories is executed with the function `fpn_end2end_train_test.py`. Its
 input argument is the path to configuration file, containing all the data, model and training parameters:

`python ./experiments/fpn_end2end_train_test.py --cfg=./experiments/cfgs/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_8.yaml`

The datasets for the training is defined in the DATASET section of the configuration .yaml file. In the config file present in the command above, two datasets (PascalVOC;ImageNet) are used, and all related fields ahave two corresponding values, separated by `;`  
The restriction of the model to a subset of all the classes is enabled via the `DATASET.cls_filter_files` argument, where pathes to files with class name-to-id LUT and the list of selected classes are provided.  
The code can use multiple GPUs. List of their ordinals is given in the 'gpus' argument in the `.yaml` file.

## Options and features of this code package
Please refer to the `parse_args()` routine in the  `few_shot_benchmark.py` for explanation on the various options available for execution.

## Data structures
### Structure of roidb
The roidb object, loaded from file `data/Imagenet_LOC/voc_inloc_gt_roidb.pkl` is a list of entries corresponding to set of images. Each entry is a dictionary with following fields:
- entry[‘gt_classes’] – list of class indices present in the image, a subset of [1,2,…1000] with possible repititions (the 1000 classes of imagenet). List of corresponding class names is given in data/Imagenet_LOC/inloc_classes_list.txt
- entry[‘image’] – full path to the image. In the provided roidb files, the image pathes need to be replaced with those available at the location of benchmark deployment.
- entry[‘boxes’] – a numpy array of bounding boxes, where the rows contain four box coordinates (left, top, right, bottom) and the rows are ordered correspondingly to the list of classes entry[‘gt_classes’]

### Structure of episodic data
According to the concept of meta-learning, the training and evaluation of a few-shot detection engine is performed using subsets of the given large datasets, known as tasks or episodes.
Each episode is an instance of a few-shot task that is comprised from a support set and query set, and contains data from Nway visual categories, each represented by Nshot examples (ROIs in the support set images). The query (evaluation) data consists of Nquery images per category, each containing one or more examples from this category (and possibly instances of other support set categories). The test data for the benchmark consists of Nepisodes such episodes, randomly drawn from the list of visual categories and images not seen during the offline training of the base model.
The episodic data (in particular, that of RepMet paper benchmark) for varying experiments is stored in the folder data/Imagenet_LOC. Each episode data file contains the list of episode objects, where each episode is a dictionary with the following fields:
episode['epi_cats'] – set of Nway class indices randomly picked from the dataset list


episode['epi_cats_names'] – corresponding set of class names (strings)  
episode[' train_nImg'] – list of image indices for training     
episode[' query_images'] – list of paths to query images of the episode     
episode[' query_gt'] – list of roidb entries corresp. to the query images   
	
## License
Copyright 2019 IBM Corp.
This repository is released under the Apachi-2.0 license (see the LICENSE file for details)	
## References:
[1] Leonid Karlinsky, Joseph Shtok, Sivan Harary, Eli Schwartz, Amit Aides, Rogerio Feris, Raja Giryes, Alex M. Bronstein, RepMet: Representative-based metric learning for classification and one-shot object detection. Accepted to CVPR 2019. https://arxiv.org/abs/1806.04728


