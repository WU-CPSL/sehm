# Self-explaining Hierarchical Model for Intraoperative Time Series

This is the repository for Self-explaining Hierarchical Model for Intraoperative Time Series (SEHM). SEHM comprises kernelized local attention and recurrent layers to effectively learn progression dynamics of high-resolution time series. A linear approximation module is parallel to the recurrent layers to ensure end-to-end interpretability. We performed evaluations on our in-house complication prediction dataset and public [HiRID](https://physionet.org/content/hirid/1.1.1/) dataset.

The manuscript is now available on [arXiv](https://arxiv.org/abs/2210.04417) and is accepted by ICDM 2022 as a short paper.

## Prerequisites

SEHM is developed with the following libraries
- Python 3.7
- PyTorch >= 1.12.1

and pip dependencies
- sklearn
- pytorch-fast-transformers>=0.3.0
- einops

## Usage

train.py is a demo script to train SEHM for prediction tasks. Please modify data_loader_handoff.py or data_loader_handoff_generator.py to load custom data and labels. An example of training SEHM is 

`python train.py --epochs 10 --model gru_sehm_kernelized --hid_size 64`

main_performance.py and eval_sehm.py are used to generate evaluation results of predictive performance and interpretability.
