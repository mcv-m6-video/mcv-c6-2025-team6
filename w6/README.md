# C6 - Team 6 - week 6

This repository is based on ['CVMasterActionRecognitionSpotting'](github.com/arturxe2/CVMasterActionRecognitionSpotting), the modify files are:

- main_spotting_transformer.py : includes the experiments related to transformer.
- model_spotting_transformer.py: includes different configuration of models to run transformer experiments.
- main_spotting.py: Extended the baseline by adding early stopping, and saving both the best and last epoch model checkpoints during training. Inference is performed on both checkpoints to compare generalization. Best model is selected based on validation loss. Training losses are also logged for monitoring.
- model_spotting.py: intilalize different model condfiguration dpending on the setting in the config.json file. 

Ensure that all files are placed correctly to maintain the expected functionality.
