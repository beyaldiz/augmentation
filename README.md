# augmentation
A PyTorch based framework for Data Augmentation to Improve Robustness of DNNs

## Running
The experiments can be conducted with the following command and all the experiment configurations are handled by config file (check `configs/`).
```
python main.py configs/{config_file}.json
```
Agent is chosen from the config file in the argument and corresponding agent's `run()` procedure is called.
After `run()` procedure is done, agent's `finalize()` procedure is called.

## Agents
All the models, datasets, procedures are handled by agents (check `agents/`).

## DL Models
Custom deep learning models can be implemented in `models/dl_models/`.

## GA Models
GA models can be implemented in `models/ga_models`, check the example GA models for more information.

## Datasets
All the datasets must be the type `augmentable` to run properly with the ga_model, check the `datasets/augmentable.py` for more information.

## Extras
- [PyTorch Template](https://github.com/moemen95/Pytorch-Project-Template)
