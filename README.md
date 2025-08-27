# ETINN
Cognitive Mimetics for Molecular Electronic Structure

## Table of Contents

1. [Introduction](#introduction)   
2. [Dependencies](#setup)      
3. [Quick start](#quickstart)   
4. [Data](#data)

## Introduction <a name="introduction"></a>
The typical gaze trajectories presented in the figure below illustrate the subject's cognitive process regarding the electronic structure of D-A structured TADF material molecules. They depict the subject's visual information acquisition concerning the donor and acceptor moieties, the bridging groups connecting them, key modifying substituents, and the subsequent information synthesis within the HOMO distribution regions.

During molecular observation, participants exhibited numerous fixations and saccades. A fixation refers to an aggregation during stabilization periods ≥200 ms, while saccades represent rapid eye movements between fixations (30-80 ms). We identified a distinctive visual behavior characterized by significant velocity reduction when visual attention traversed specific areas of interest. These areas consistently showed directional deviations in gaze movement or repeated back-and-forth movements in gaze paths. We term these regions Decelerated Saccade Zones (DSZs). Experimentally, we observed that DSZs primarily occurred in regions connecting donor (D) and acceptor (A) groups, as well as their critical modification sites. Fixations and DSZs were consistently occurred across all trials, indicating their fundamental role in participants' cognition of FMOs. Temporal analysis revealed distinct attentional phases. Based on the analysis results, we established the ETINN model to reproduce their multi-stage feature processing for a better understanding of molecular electronic structure.


## Dependencies <a name="setup"></a>
After git cloning the repository, the environment that the ETINN model depends on is as follows:

* OS support: Linux
* Python version: 3.8.18

| name         | version |
| ------------ | ---- |
| numpy        | 1.22.4 |
| pandas       | 2.0.3 |
| networkx     | 3.1 |
| torch | 2.0.1 |
| scikit-learn      | 1.1.0 |
| matplotlib      | 3.4.3 |
| ase             | 3.22.1 |

## Quick start <a name="quickstart"></a>
for the training and testing the method in the paper:
follow the steps below in the folder ETINN 

1 Training: run python ETINN_main.py --data path_to_xyz_files --labels path_to_labels --cutoff 4 --cutoff_prob 4 --num_interactions 4 --split_file ./data/datasplits.json --output_dir  path_to_output

2 Testing: run python ETINN_eval.py path_to_your_model --data path_to_xyz_files --labels path_to_labels

3 Check saliency maps: run python ETINN_eval_check_attn.py 'obj-mol-name' path_to_your_model --labels path_to_labels --data path_to_xyz_files --output_dir output_path

The script will generate images of saliency maps in the output folder.

## Data <a name="data"></a>
We collected a total of 733 published literature on TADF molecular materials, and recorded the xyz and DFT-calculated related properties of HOMO-LUMO distributions.
The data file can be found in the file folder ./data
