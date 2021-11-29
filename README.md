# FCS-applications
Source code for CsiNet and CRNet using the Fully Connected Layer-Shared feedback architecture. 
# Introduction
This repository contains the program of the training and testing procedures of FCS-CsiNet and FCS-CRNet proposed in Boyuan Zhang, Haozhen Li, Xin Liang, Xinyu Gu, and Lin Zhang, "Fully Connected Layer-Shared Network Architecture for Massive MIMO CSI Feedback" (submitted to IET Electronics Letters).
# Requirements
- Python 3.5 (or 3.6)
- Keras (>=2.1.1)
- Tensorflow (>=1.4)
- Numpy
# Instructions
The following instructions are necessary before the network training:
- The repository only provide the programs used for the training and testing of the FCS-CsiNet and FCS-CRNet in the form of python files. The network models in the form of h5 files are not included.
- The part "settings of GPU" in each python file should be adjusted in advance according to the device setting of the user.
- The folds named "result" and "data" should be established in advance in the folds "FCS-CsiNet" and "FCS-CRNet" to store the models obtained during the training procedure and to store the dataset used for training and testing.
- The dataset used in the network training can be downloaded from https://drive.google.com/drive/folders/1_lAMLk_5k1Z8zJQlTr5NRnSD6ACaNRtj?usp=sharing, which is first provided in https://github.com/sydney222/Python_CsiNet). The dataset should be put in the folds "data".
Therefore, the structure of the folds "FCS-CsiNet" and "FCS-CRNet" should be:
```
*.py
result/
data/
  *.mat
```
# Training Procedure 
The training and testing procedures are demonstrated as follows:
## Step.1 Main training process
Run Step1_main_training_1.py and Step1_main_training_12.py to obtain the parameters of the shared FC layer and the pre-trained models of the other parts of the network.
## Step.2 Assistant review processes
Run Step2_assistant_review.py to obtain the model used in Scenario_1. The feedback accuracy of the model in Scenario_1 will be also be calculated in Step.2.
## Step.3 Assistant compensation process
Run Step3_assistant_compensation.py to obtain the model used in Scenario_2. The feedback accuracy of the model in Scenario_2 will be also be calculated in Step.3.

The results are given in the submitted manuscript "Fully Connected Layer-Shared Network Architecture for Massive MIMO CSI Feedback".
