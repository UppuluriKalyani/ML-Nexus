# Content-Based Movie Recommendation System

## Overview

This project implements a content-based recommendation system using plot summaries from movies. Given a movie title, the system recommends similar movies based on the similarity of their plot summaries.

## Dataset

The project uses two datasets:
- `tmdb_5000_credits.csv`: Contains information about the cast and crew of the movies.
- `tmdb_5000_movies.csv`: Contains metadata about the movies, including title, overview, genres, and other relevant details.

## Features

- Data cleaning and merging of credits and movies datasets.
- Extraction of features from the movie overview using TF-IDF Vectorization.
- Calculation of movie similarity using the sigmoid kernel.
- Recommendation of similar movies based on user input.

## Getting Started

### Prerequisites

Make sure you have the following libraries installed:

- pandas
- numpy
- scikit-learn

You can install them using pip:

```bash
pip install pandas numpy scikit-learn
```

## Usage
- Load the Datasets: Load the tmdb_5000_credits.csv and tmdb_5000_movies.csv datasets.
- Data Cleaning and Merging: Clean and merge the datasets to create a unified DataFrame.
- Create Recommendations: Use the give_recommendations function to find similar movies based on a given title.

## Function Definitions
give_recommendations(title, sig=sig)
- Parameters:
    - title (str): The title of the movie for which recommendations are to be found.
    - sig (array-like): The similarity matrix computed from the TF-IDF matrix.
- Returns: A list of recommended movie titles based on the provided title.
## Conclusion
This content-based movie recommendation system provides a way to explore movies based on their plot summaries. It can be further enhanced by integrating additional features like user ratings or incorporating other recommendation techniques.
