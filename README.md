# Feedback GAN for Enhancer

## System Requirements and Installation
```pip install -r requirements.txt```
- Python 3.6.3

## Demo Instructions
All default arguments for demo are provided.

1) Run `python wgan_gp_gene` to pretrain WGAN with Gradient Penalty to produce valid enhancer sequences.
  - **Expected Output**
    - `sample/$RUN_NAME` will contain sample gene sequences from every 100 iterations, as well as loss and distance curves.
    - `checkpoint/$RUN_NAME` will contain checkpoints for both generator and discriminator

2) Train FBGAN:
  - With DeepSTARR analyzer: run `python wgan_gp_lang_gene_analyzer`. Default arguments for demo are provided.
  - **Expected Output**
    - `checkpoint/$RUN_NAME` will contain FBGAN checkpoints
    - `samples/$RUN_NAME/sampled_*_preds.txt` will contain sampled outputs from generator and their scores from analyzer from every epoch.
    - $RUN_NAME for DeepSTARR demo is default "fbgan_amp_demo"
  -**Expected Runtime**
    - With a Nvidia GEFORCE GTX GPU, it takes roughly 2 minutes per epoch for FBGAN. Expect this to take at least 10 minutes per epoch on a desktop.

## LICENSE
  Source files are made available under the terms of the GNU Affero General Public License (GNU AGPLv3). See GNU-AGPL-3.0.txt for details.
