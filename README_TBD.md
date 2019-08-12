# RepMet
Few-shot detection benchmark and the RepMet detection engine

## Introduction
This manual describes the python code package implementing two modules:
1. Few-shot detection benchmark 
   - Produce episodic data
   - Manage a detection engine

2. RepMet detection engine
   - Train a model for episode data (novel categories) based on pretrained model
   - Perform detection and perfromance evaluation

The RepMet algorithm is described in the paper [1] 
	
	
	
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

requirements.txt can be found in the root of the repository (RepMet/), or created as a text file with following lines:
Cython
EasyDict
opencv-python
mxnet-cu80==1.0.0

The dataset, models and the rest of the data are originally located on /dccstor storage, used at the CCC computation cluster. All the relevant files are transferred to the folder ‘data’ in the IBM BOX folder https://ibm.box.com/s/49sy4aordtt3063shb50c1nze1jzh4ow . For deployment, this folder should be placed under the RepMet root of the git package. 

The dataset information (images, Ground Truth boxes and classes) is given in a roidb_*.pkl file, produced during the training of the base model. The structure of roidb is described below. In order to use the package with other datasets, a code for creating one may be based on `RepMet/lib/dataset/imagenet.py`

Before the benchmark can be executed, all the image paths in the roidb need to be replaced; all the rest of the paths are in the format `./data/…`, they should work once the Box content is copied under the repository root `RepMet/`. 


## Example execution of a benchmark experiment

From the main folder of the repository, `/RepMet`, execute 

`python fpn/run_FSD_benchmark_v1.py --bcfg_fname=./experiments/bench_configs/Inloc_RepMet_3_6_2_5.yaml`

The run script run_FSD_benchmark_v1.py creates an instance of FSD_bench class with given arguments, 
benchmark = FSD_bench(args) and runs its following routines:

benchmark.setup()
benchmark.gen_episodes() # prep training data, setup detector
benchmark.run_episodes()

The setup() routine initiates the experiment.
The gen_episodes() routine loads the set of episodes for the benchmark, or generates the episodes with the bcfg parameters if the episodes file does not exist.
The run_episodes() routine creates a FSD_engine class instance to train a model for each episode and runs a test on the query images for this episode.
Expected output for this experiment:

'2019-03-29 13:53:01,336 - my_logger - INFO -  ========= starting FSD benchmark Inloc_RepMet_3_6_2_5
2019-03-29 13:53:11,092 - my_logger - INFO - Creating new episodes data.

2019-03-29 13:53:23,588 - my_logger - INFO - Starting episode 0 -------
2019-03-29 13:54:08,492 - my_logger - INFO - _ #Dets: 16570, #GT: 15 TP: 15 FP: 16555 = 75 wrong + 16480 bkgnd  Recall: 1.000 AP: 0.852

2019-03-29 13:54:08,493 - my_logger - INFO - Starting episode 1 -------
Box #0 - no overlapping detection boxes found
2019-03-29 13:54:18,727 - my_logger - INFO - No rois found for training box of eskimo dog, husky
2019-03-29 13:54:46,936 - my_logger - INFO - _ #Dets: 30426, #GT: 30 TP: 30 FP: 30396 = 150 wrong + 30246 bkgnd  Recall: 1.000 AP: 0.654

2019-03-29 13:54:46,937 - my_logger - INFO - Starting episode 2 -------
2019-03-29 13:55:24,407 - my_logger - INFO - _ #Dets: 45216, #GT: 45 TP: 43 FP: 45173 = 217 wrong + 44956 bkgnd  Recall: 0.956 AP: 0.606

2019-03-29 13:55:24,407 - my_logger - INFO - Starting episode 3 -------
Box #0 - no overlapping detection boxes found
2019-03-29 13:55:32,141 - my_logger - INFO - No rois found for training box of king snake, kingsnake
2019-03-29 13:56:03,580 - my_logger - INFO - _ #Dets: 61764, #GT: 58 TP: 56 FP: 61708 = 282 wrong + 61426 bkgnd  Recall: 0.966 AP: 0.631

2019-03-29 13:56:03,580 - my_logger - INFO - Starting episode 4 -------
2019-03-29 13:56:41,246 - my_logger - INFO - _ #Dets: 76367, #GT: 73 TP: 71 FP: 76296 = 350 wrong + 75946 bkgnd  Recall: 0.973 AP: 0.565'


## Files in the data folder (in Box repository):
All the available files are related to the Imagenet-LOC dataset, hence are placed under the `/data/Imagenet_LOC/` folder

| File 								| Information |
| --- | --- |
|inloc_first101_categories.txt		| List of all the classes in Imagenet-LOC dataset |
|in_domain_categories_ord.txt		| List of the classes selected for few-shot experiments, along with their original indices in the total list |
|voc_inloc_gt_roidb.pkl				| The roidb database containing ground truth information. |
|RepMet_pascal_imagenet-0015.params	| Model weights for RepMet detection model |
|Episodes/epi_inloc_in_domain_1_5_10_500.pkl | (and similar ones)	An episodic data file for benchmark |


Replacement of image pathes:
The list entries from voc_inloc_gt_roidb.pkl have a field ‘image’ (see section ‘Structure of roidb for rull information). The replacement needs to be performed as follows:

| Current value 	| New value |
| --- | --- |
| roidb[i][‘image’] = <old path>/<image_name.jpg> | roidb[i][‘image’] = <new path>/<image_name.jpg> |

where the new value depends on the Imagenet-LOC dataset deployment.
The experiments, described in RepMet CVPR paper [1] may be repeated using the corresponding bcfg config files and stored episodic data.
Execute python fpn/run_FSD_benchmark_v1.py -- bcfg_fname=<bench config fname>
with the following values of <bench config fname>:


| Experiment 	| bcfg_fname |
| --- | --- |
|RepMet, 3-shot, 6-way	| ./experiments/bench_configs/Inloc_RepMet_3_6_2_5.yaml |
|RepMet, 1-shot, 5-way	| ./experiments/bench_configs/Inloc_RepMet_1_5_10_500.yaml |
|RepMet, 5-shot, 5-way	| ./experiments/bench_configs/Inloc_RepMet_5_5_10_500.yaml |




## Data structures
### Structure of roidb
The roidb object, loaded from file `data/Imagenet_LOC/voc_inloc_gt_roidb.pkl` is a list of entries corresponding to set of images. Each entry is a dictionary with following fields:
- entry[‘gt_classes’] – list of class indices present in the image, a subset of [1,2,…1000] with possible repititions (the 1000 classes of imagenet). List of corresponding class names is given in data/Imagenet_LOC/inloc_classes_list.txt
- entry[‘image’] – full path to the image. In the provided roidb files, the image pathes need to be replaced with those available at the location of benchmark deployment.
- entry[‘boxes’] – a numpy array of bounding boxes, where the rows contain four box coordinates (left, top, right, bottom) and the rows are ordered correspondingly to the list of classes entry[‘gt_classes’]

### Structure of episodic data
According to the concept of meta-learning, the training and evaluation of a few-shot detection engine is performed using subsets of the given large datasets, known as tasks or episodes.
Each episode is an instance of a few-shot task that is comprised from a support set and query set, and contains data from Nway visual categories, each represented by Nshot examples (ROIs in the support set images). The query (evaluation) data consists of Nquery images per category, each containing one or more examples from this category (and possibly instances of other support set categories). The test data for the benchmark consists of Nepisodes such episodes, randomly drawn from the list of visual categories and images not seen during the offline training of the base model.
The episodic data (in particular, that of RepMet paper benchmark) for varying experiments is stored in the folder data\Imagenet_LOC. Each episode data file contains the list of episode objects, where each episode is a dictionary with the following fields:
episode['epi_cats'] – set of Nway class indices randomly picked from the dataset list

episode['epi_cats_names'] – corresponding set of class names (strings)
episode[' train_boxes'] – list of train objects. Each object is represented by the list 
	train_box = [class_index, class_name, image_path, bounding_box]
episode[' query_images'] – list of paths to query images of the episode

episode[' query_gt'] – list of roidb entries corresp. to the query images.

For other datasets or new benchmarks, the set of episodes can also be generated using the routine FSD_bench.gen_episodes().


### Structure of input parameters
The benchmark class FSD_bench has one required and two optional parameters.

#### bcfg_fname
(necessary parameter)  is a path to the benchmark configuration (bcfg) file. A fast working example is RepMet/experiments/bench_configs/Inloc_RepMet_3_6_2_5.yaml. The name format is <dataset>_<detection engine>_<Nshot>_<Nway>_<Nquery>_<Nepisodes>.yaml (see the meaning of the variable names in “structure of episodic data” section of this manual).
The bcfg file for a specific experiment contains only a few fields overriding the values in the full bench_config.py set of parameters. So all the default values of the benchmark configuration are given in RepMet/fpn/config/bench_config.py
The detection engine RepMet has its own main configuration file  RepMet/fpn/config/config.py and a model-specific update for this configuration, which path is provided as a parameter in the bcfg. The detection engine configuration is loaded within the FSD_engine class.

#### bcfg_args
(optional) is a command line argument string containing numeric confuguration parameters (from either benchmark or detection engine configuration files) that shall override any values in the default configuration files (e.g., RepMet/fpn/config/bench_config.py or RepMet/fpn/config/config.py) and in the experiment-specific yml files (e.g., experiments/bench_configs/Inloc_RepMet_3_6_2_5.yaml or ./experiments/fpn/cfgs/oneshot/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_8_dev.yaml). 
An example for bcfg_args value:
--bcgf_args=”TRAIN.lr=0.002, detector.nms_train=0.4”
The first parameter belongs to detection engine (dcfg.TRAIN.lr = learning rate), and the second happens to be from benchmark configuration (NMS value for train object rois selection).

#### gpu
(optional) – the index of allocated GPU number (default=0). Relevant for multi-GPU servers.



	
	
## References:
[1] Leonid Karlinsky, Joseph Shtok, Sivan Harary, Eli Schwartz, Amit Aides, Rogerio Feris, Raja Giryes, Alex M. Bronstein, RepMet: Representative-based metric learning for classification and one-shot object detection. Accepted to CVPR 2019. https://arxiv.org/abs/1806.04728


