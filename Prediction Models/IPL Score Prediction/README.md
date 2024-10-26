# IPL Score Predictor using Deep Learning

## Overview

This project aims to predict the score in an IPL cricket match based on various inputs using deep learning techniques. The inputs include the venue of the match, the batting team, the bowling team, the current striker, the bowler, and historical match data. The model utilizes a deep learning architecture to learn patterns from past matches and generate a predicted score for the ongoing match.

## Features

- **Input Features:**
    - Venue: The stadium where the match is being played.
    - Batting Team: The team currently batting.
    - Bowling Team: The team currently bowling.
    - Striker: The batsman currently facing the delivery.
    - Bowler: The bowler delivering the ball.

- **Output:**
    - Predicted score: The estimated total score that the batting team is likely to achieve based on the given inputs.

## Data Collection

The model relies on a comprehensive dataset of historical IPL matches. This dataset includes information about venues, participating teams, players, and match outcomes. The data is preprocessed and used to train the deep learning model.

## Model Architecture

The deep learning model utilizes a combination of neural network layers, including standard feedforward neural network. The model is trained using historical match per ball data, including the aforementioned inputs and actual match scores.

## Usage

To use the IPL score predictor:

1. **Input Preparation:**
    - Gather the required information: venue, batting team, bowling team, striker, and bowler for the ongoing match.
    - Ensure the input data is in the required format or can be encoded appropriately for the model.

2. **Prediction:**
    - Feed the input data into the trained model.
    - Obtain the predicted score as the output.

## Getting Started

To set up the environment and run the predictor:

1. Clone the repository:

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Run the predictor:

   ```
   python predict_score.py --venue "VenueName" --batting_team "TeamName" --bowling_team "OpponentTeam" --striker "StrikerName" --bowler "BowlerName"
   ```

## Example

An example of using the predictor:

```bash
python predict_score.py --venue "Eden Gardens" --batting_team "Kolkata Knight Riders" --bowling_team "Mumbai Indians" --striker "Shubman Gill" --bowler "Jasprit Bumrah"
```

This would output the predicted score for the given match scenario.