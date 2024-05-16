# Mixture of Depths

### Directory Structure
```
└───MoD
    ├─── mixture_of_depths (Contains all the modelling/Architecture and Training Code)
        ├───routing_transformer.py (Contains the main model code)
    │   └───generation.py (Contains the code to sample from the model)
    ├─── slurm (Contains Sbatch files to run)
    ├─── profiling_logs (Contains all the profiling logs and code for processing and extracting info from the logs)
    ├─── utils (Dataset Analysis Utils)
    ├─── MoD_sampling.py (Code to sample from the model)
    ├─── MoD_training.py (Code to train the model, with required args)
    ├─── perplexity.py (File contains the code to measure perplexity of the trained models)
    └─── simulate.py (Code to simulate the scheduling policies)    
```

### Model Training
To Train the model, run
```
python MoD_training.py --epochs n --dim d --aux_routing True ...
```

### Model Sampling
To Sample from the model, run
```
python MoD_sampling.py
```
The prompts can be modified inside the file.
The pretrained models weights are available at : https://drive.google.com/file/d/1N8B2abpNb72Zh9F0FrNHb3nJFb75nOuk/view?usp=sharing

### Inference Simulation
To simulate the inference scheduling policies, run:
```
python simulate.py
```