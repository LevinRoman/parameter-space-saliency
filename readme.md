## How to use

### 1. create a conda environment
```
conda create -n lama37 -y python=3.7 && conda activate lama37
pip install -r requirements.txt
```
### 2. download the data
```
cd /your/path/to/the/dataset
wget https://dl.fbaipublicfiles.com/LAMA/data.zip
unzip data.zip
rm data.zip
```

### 3. Run the nearest neighbor search in the Squad dataset
```
python knn.py --dataset_name /your/path/to/the/datasets/data/Squad/test.jsonl 
```
Other args:     
* `k_closest`: the number of NNs to be shown; 
* `partial`: comparing only the partial saliency profile(deeper or shallower stats).        
* See the `help` entry in `parser.add_arguments` for more details. 

The default value of `chosen_samples` is empty, in which case 6 random samples will be chosen, controlled by a random seed `args.seed`. 


### *Other datasets*
Currently, I've only tested the code on **Squad** and **ConceptNet**. For a quicker run, use the **Squad** dataset, which only has ~300 samples. **ConceptNet** has ~30,000 samples, and the computation of saliency profiles can take about 30 minutes to finish. The pattern in the nearest neighbors may be more representative using a larger sample size. 
