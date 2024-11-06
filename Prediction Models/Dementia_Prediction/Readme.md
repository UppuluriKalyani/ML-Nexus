# Dementia Prediction Model

This repository contains a machine learning model that predicts dementia in patients based on medical and demographic data. The model utilizes various supervised learning techniques to analyze historical patient data and provides a probability score indicating the likelihood of dementia diagnosis.

## Table of Contents
- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Data](#data)


## Introduction

Dementia is a chronic condition that affects cognitive function, leading to memory loss and impaired reasoning. Early prediction of dementia can facilitate timely intervention and improve patient care. This project uses machine learning models to predict dementia based on patient data, such as age, cognitive test results, and medical history. 

## Problem Statement

Dementia is challenging to predict due to its complex and progressive nature. Several factors contribute to the risk of dementia, including:
- **Demographic Information**: Age, gender, education level.
- **Medical History**: Family history of dementia, comorbid conditions.
- **Cognitive Test Scores**: Results from standardized assessments.

The primary challenge is to develop a model that accurately predicts dementia by identifying patterns in patient data.

## Solution Overview

The model is trained using various machine learning algorithms to determine the most effective approach for dementia prediction. Algorithms include:
- **Logistic Regression**
- **Decision Trees**
- **Random Forests**
- **Support Vector Machines (SVM)**

The model uses supervised learning and is trained on a dataset of patient records containing features such as age, cognitive test scores, and medical history.

## Data

The dataset includes:
- **Patient Demographics**: Age, gender, education level.
- **Cognitive Scores**: Scores from memory and cognitive assessments.
- **Medical History**: Previous diagnoses, family history, and comorbid conditions.
  
Data should be placed in the `data/` folder in CSV format for model training and evaluation.


