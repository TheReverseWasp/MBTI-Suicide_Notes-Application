# Myers-Brigs Type Indicator Transformer approaches to detect Suicide Notes

This repository is based on the simple statement of "prove MBTI can detect patterns in psychology's scenarios" in this case suicide. We use two datasets included MBTI Kaggle and Suicide Notes Dataset. We want to replicate Janowsky et al. 2002 study about the relationship between suicide and MBTI.

We compared several transformer models in a way to achieve this goal we follow these steps:

# Training

We train/test and evaluate the transformer models between each other using the same conditions for all of them.

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

![Untitled](Myers-Brigs%20Type%20Indicator%20Transformer%20approaches%20%206dcbc04d439247fcb5fdfe0e9b94f3a4/Untitled.png)

and AUC (%):

![Untitled](Myers-Brigs%20Type%20Indicator%20Transformer%20approaches%20%206dcbc04d439247fcb5fdfe0e9b94f3a4/Untitled%201.png)

# Achieved Suicide Notes Dataset Results

The following image explains the achieved results by each model in the Suicide Notes dataset. we focus in three measures to get the best model:

- High F1/AUC Score
- Blue Accuracy:

$$
blue\_accyracy = \frac{blue\_personalities}{all\_personalities}
$$

- Green Accuracy:

$$
green\_accuracy =  \frac{(blue\_personalities + green\_personalities)}{ all\_personalities}
$$

These measures are made based in Janowsky et al MBTI Research over Suicide Patients.

![Untitled](Myers-Brigs%20Type%20Indicator%20Transformer%20approaches%20%206dcbc04d439247fcb5fdfe0e9b94f3a4/Untitled%202.png)

# Architecture

We used as a setup a processor i5-10400F with an Ubuntu 21.10 as operating system with 16 GB of RAM memory and a GeForce RTX 3060 with 12 GB of graphics memory for the experiments.

# Replication

To execute the experiments you need to follow the next steps:

```
######################## Installation ###############################

# 1. Create a Virtual Enviroment
conda create -n mbti_sn python=3.8

# 2. Activate the Environment
conda activate mbti_sn

# 3. Install Dependencies
pip install notebook==6.4.10
pip install seaborn==0.11.2

# You can use pytorch 1.12 or 1.13
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install simpletransformers==0.61.4
pip install nltk==3.7
pip install wordcloud==1.8.1
pip install statsmodels==0.13.2
pip install scikit-learn==1.1.2

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
# 1st paper

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
# Thesis
@article{lazo2022clasificacion,
  title={Clasificaci{\'o}n de la personalidad utilizando procesamiento de lenguaje natural y aprendizaje profundo para detectar patrones de notas de suicidio en redes sociales},
  author={Lazo Vasquez, Ricardo Manuel},
  year={2022},
  publisher={Universidad Cat{\'o}lica San Pablo}
} 

# 2nd paper
#### Pending
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