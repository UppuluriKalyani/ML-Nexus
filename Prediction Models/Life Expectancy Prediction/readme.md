# Life Expectancy Prediction System

## Overview

This project aims to develop a data-driven system to analyze and predict life expectancy across 193 countries using a comprehensive dataset spanning from 2000 to 2015. The system focuses on identifying significant predictors of life expectancy, including immunization rates, GDP, schooling, and health expenditure, with the goal of guiding policy decisions for health improvements.

## Problem Statement

Despite extensive research on factors affecting life expectancy, key elements like immunizations and human development indices have often been overlooked. Additionally, many studies have relied on single-year datasets, limiting their effectiveness. This project seeks to address these gaps by utilizing a mixed-effects model and multiple linear regression based on a multi-year dataset.

## Objectives

- Conduct exploratory data analysis (EDA) to understand data distributions and relationships.
- Perform feature importance analysis to identify the most significant predictors of life expectancy.
- Develop predictive models to forecast life expectancy based on selected health and economic factors.

## Data Sources

The dataset used in this project has been sourced from:

- **World Health Organization (WHO)**: Global Health Observatory (GHO) data repository.
- **United Nations (UN)**: Economic data and related health factors.

- **Immunization-related factors** (e.g., Hepatitis B, Polio, Diphtheria)
- **Mortality factors** (e.g., adult mortality, under-5 deaths)
- **Economic factors** (e.g., GDP, health expenditure)
- **Social factors** (e.g., schooling)

## Methodology

1. **Data Preparation**: The data was cleaned to handle missing values, particularly for variables like population, Hepatitis B, and GDP. Countries with significant missing data were excluded.
2. **Exploratory Data Analysis (EDA)**: Visualizations and statistical analyses were performed to uncover patterns and correlations.
3. **Feature Importance Analysis**: The significance of different predictors was assessed using methods like Random Forests.
4. **Predictive Modeling**: Models such as mixed-effects models and multiple linear regression were developed to predict life expectancy.
