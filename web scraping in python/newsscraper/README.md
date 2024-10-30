# News Scraper

A Python-based web scraper designed to extract news articles from various sources, analyze their sentiment, and provide summaries of the content. This project leverages Selenium for web scraping and advanced Natural Language Processing (NLP) techniques, including Gemini AI for sentiment analysis and summarization.

## Features

- **Web Scraping**: 
  - Efficiently extracts articles from Google News and other news websites.
  - Retrieves full content by following article links.

- **Sentiment Analysis**: 
  - Merges titles and content of articles for comprehensive analysis.
  - Utilizes Gemini AI for advanced sentiment analysis, providing accurate sentiment scores (positive, negative, neutral).

- **Summarization**:
  - Employs Gemini AI to generate concise summaries of the articles, helping users quickly grasp the main points.

- **Data Storage**: 
  - Organizes and saves extracted data in structured formats, such as CSV or JSON.

## Technologies Used

- Python
- Selenium
- Pandas
- BeautifulSoup
- Gemini AI for sentiment analysis and summarization
- Natural Language Processing (NLP) libraries, such as VADER and TextBlob

## Installation

To get started with the News Scraper, clone the repository and install the required libraries. Ensure that you have the appropriate WebDriver for your browser installed and configured.

## Usage

After setting up the environment and installing the necessary libraries, you can run the scraper. Modify the configuration files to specify your desired news sources, and execute the script to start scraping articles. The results, including the sentiment analysis and summaries, will be saved in a specified output file.

## Example Output

The output will include a structured table with the following columns:

- Title
- Link
- Description
- Content
- Published Date
- Source
- Sentiment Analysis Result
- Summary


## Acknowledgments

- Selenium for web automation
- Pandas for data manipulation
- BeautifulSoup for HTML parsing
- Gemini AI for advanced sentiment analysis and summarization
- Natural Language Toolkit (NLTK) for text processing
