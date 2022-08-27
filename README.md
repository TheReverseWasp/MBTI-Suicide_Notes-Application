# Myers-Brigs Type Indicator Transformer approaches to detect Suicide Notes

This repository is based on the simple statement of "prove MBTI can detect patterns in psychology's scenarios" in this case suicide. We use two datasets included MBTI Kaggle and Suicide Notes Dataset. We want to replicate Janowsky et al. 2002 study about the relationship between suicide and MBTI.

We compared several transformer models in a way to achieve this goal we follow these steps:

# Training

We train/test and evaluate the transformer models between each other using the same conditions for all of them specified in the next Table:

| Name | Value |
| --- | --- |
| Sentence max size (token count) | 512 |
| Epochs | 15 |
| Batch Size | 3 |
| Learning Rate | 5e-5 |
| Early Stopping Patience | 3 |
| Initial Seed | 4 |
| Train/test segments | 90/10 |
| Train/test data | The same for all models |

# Used Models

We evaluated the next list of Transformer models:

1. BERT Base.
2. BERT Multilingual.
3. Distil BERT Base.
4. Distil BERT Multilingual.
5. ELECTRA Base.
6. ELECTRA Small.
7. RoBERTa Base.
8. RoBERTa Distil.
9. XLM-RoBERTa.
10. XLNet Base.

## Achieved MBTI Kaggle Results

We achieve the following results over F1-Score (%).

| F1 - Score in % |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Model | IE | NS | TF | JP | Average |
| BERT - Base | 91.1095 | 93.637 | 82.5032 | 72.4595 | 84.9273 |
| BERT - Multilingual | 89.368 | 94.6322 | 83.0769 | 73.7313 | 85.2021 |
| Distil BERT Base | 91.6728 | 94.5333 | 82.4934 | 79.8246 | 87.131 |
| Distil BERT Multilingual | 90.1325 | 94.4704 | 82.9706 | 74.344 | 85.4794 |
| XLNet - Base | 92.3414 | 95.3826 | 85.2288 | 76.6197 | 87.3931 |
| RoBERTa - Base | 88.8778 | 92.4401 | 0 | 0 | 44.8295 |
| RoBERTa - Distil | 92.5571 | 95.1123 | 85.3437 | 78.4483 | 87.8653 |
| XLM - RoBERTa | 86.8778 | 92.4401 | 0 | 0 | 44.8295 |
| ELECTRA - Small | 92.0588 | 95.1688 | 85.1948 | 76.9231 | 87.3364 |
| ELECTRA - Base | 90.8414 | 94.6322 | 83.3987 | 78.4993 | 86.8429 |

and AUC:

| AUC in % |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Model | IE | NS | TF | JP | Average |
| BERT - Base | 85.8726 | 87.2540 | 89.2143 | 83.9744 | 86.5788 |
| BERT - Multilingual | 82.8268 | 85.0745 | 89.0668 | 85.4521 | 85.6051 |
| DistilBERT - Base | 86.9968 | 88.2249 | 91.3729 | 88.2115 | 88.7015 |
| DistilBERT - Multilingual | 84.8185 | 88.3881 | 90.9138 | 85.0513 | 87.2940 |
| XLNet - Base | 85.7396 | 87.6427 | 91.1150 | 87.0429 | 87.8850 |
| RoBERTa - Base | 47.3625 | 51.7828 | 52.0374 | 52.2463 | 50.8572 |
| RoBERTa - Distil | 86.8296 | 86.2405 | 92.0942 | 89.2766 | 88.6102 |
| XLM-RoBERTa | 49.2457 | 52.4169 | 56.1185 | 48.0545 | 51.4589 |
| ELECTRA - Small | 86.8161 | 85.9907 | 89.8888 | 85.4999 | 87.0469 |
| ELECTRA - Base | 86.7120 | 86.5751 | 89.1168 | 87.3906 | 87.4486 |

# Achieved Suicide Notes Dataset Results

The following image explains the achieved results by each model in the Suicide Notes dataset. we focus in three measures to get the best model:

1. Variety of personalities gets the Bold measure (Best 8 models).
2. INFP in the top gets the (W) symbol (Best 4 models).
3. Most INFP in the top gets (Best model) Symbol (Best model): ELECTRA - Base.

These measures are made based in Janowsky et al MBTI Research over Suicide Patients.

![Untitled](Myers-Brigs%20Type%20Indicator%20Transformer%20approaches%20%20583a4b318da04bfb8fcee8814e63e858/Untitled.png)

# Architecture

We used an Ubuntu 21.10 with 16 GB of RAM memory and a GeForce RTX 3060 with 12 GB of graphics memory for the experiments.

# Replication

To execute the experiments you need to follow the next steps:

```
######################## Installation ###############################

# 1. Create a Virtual Enviroment
conda create -n transformers python=3.8

# 2. Activate it
conda activate transformers

# 3. Install Torch
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# 4. Install Simpletransformers
pip install simpletransformers==0.61.4

######################## Execution ###############################

# 0. Go to Scripts folder
cd Scripts

# 1. Run the training
./runner.sh # Aprox 20 hours with the specified architecture

# 2. Run Suicide Prediction
./suicide_predicter.sh

# 3. Get data for ROC Curves
./roc_script.sh
```

# Citation

## First Research (Only MBTI)

```
@inproceedings{vasquez2021transformer,
  title={Transformer-based Approaches for Personality Detection using the MBTI Model},
  author={V{\'a}squez, Ricardo Lazo and Ochoa-Luna, Jos{\'e}},
  booktitle={2021 XLVII Latin American Computing Conference (CLEI)},
  pages={1--7},
  year={2021},
  organization={IEEE}
}
```

## Second Research (MBTI + Suicide)

```
#pending
```

## Janowsky et al. Work

```
@article{janowsky2002relationship,
  title={Relationship of Myers Briggs type indicator personality characteristics to suicidality in affective disorder patients},
  author={Janowsky, David S and Morter, Shirley and Hong, Liyi},
  journal={Journal of psychiatric research},
  volume={36},
  number={1},
  pages={33--39},
  year={2002},
  publisher={Elsevier}
}
```