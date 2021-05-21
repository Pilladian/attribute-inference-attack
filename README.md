# Attribute Inference Attack
Given a target machine learning model, an *Attribute Inference Attack* aims to infer
attributes of the training data. In our case we have the
target model trained to predict the gender (male, female) of a person. The attribute
we want to infer is race. Therefor we train an attacker model (1) based on the output posterior
of the target model and (2) based on the output of the last hidden layer of the target model.

## Target Model
As our target model we used a *Convolutional Neural Network* with the following hyper-parameter.

| Parameter     | Value
|------         |------
| Type          | CNN
| Convolution Layer | 2
| Linear Layer  | 4
| Epochs        | 10
| Learning Rate | 0.00001
| Dropout       | 0.5
| Optimizer     | Adam
| Loss Function | CrossEntropyLoss
| Accuracy      | 0.8863

## Attacker Model
As our attacker model we used a *Multi Layer Perceptron* with the following hyper-parameter.

| Parameter     | Value
|------         |------
| Type          | MLP
| Hidden Layer  | 1
| Epochs        | 1500
| Learning Rate | 0.01
| Dropout       | 0.5
| Optimizer     | Adam
| Loss Function | CrossEntropyLoss

## Dataset
As our dataset we used [UTKFace](https://www.kaggle.com/jangedoo/utkface-new). We sampled them into three sets (train, eval, test) and used train and eval for training and evaluating the target model and the testset for sampling the attacker dataset. For the attacker dataset we queried the target model on the testset and used the output as features.

## Attribute Inference Attacks

### Baseline
As baseline we choose the guessing baseline. Since our attacker dataset consists of 342 images for each of the 5 classes, our baseline accuracy is 0.2

### Attack 1

##### Threat Model
- **Model :** Black Box Access to target
- **Dataset :** Same distribution dataset

##### Attack Methodology
- Query target model on some image to predict the gender
- Use output posterior to train the attacker model

### Attack 2

##### Threat Model
- **Model :** White Box Access to target
- **Dataset :** Same distribution dataset

##### Attack Methodology
- Query target model on some image to predict the gender
- Use output of last hidden layer of target model to train the attacker model

### Results

| Attack   | Attackers Accuracy
|---       |---
| Baseline | 0.2
| Attack 1 | 0.3080
| Attack 2 | 0.8646
