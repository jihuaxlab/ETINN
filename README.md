# ETINN
Cognitive Mimetics for Molecular Electronic Structure

## Table of Contents

1. [Dependencies](#setup)      
2. [Quick start](#quickstart)   
3. [Data](#data)  

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
