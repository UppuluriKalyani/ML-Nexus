# WebWeaver: AI Web Scraper

Welcome to **WebWeaver**, an AI-powered web scraping and content parsing tool. This project allows users to input a website URL, scrape its content, and parse specific information using OpenAI’s language models. The app is built with Streamlit to provide a simple and interactive web interface.

## Features
1. **Web Scraping**: 
   - Users can input a URL, and WebWeaver scrapes the content of the web page using Selenium.
   - Extracts only the textual content from the web page, removing scripts and styling for cleaner output.

2. **Content Display**: 
   - After scraping, the app displays the website’s clean text content for users to review.

3. **AI-Powered Parsing**:
   - Users can provide a description of what specific information they want to extract from the web page content.
   - Using OpenAI’s API, WebWeaver processes the content in chunks and returns only the relevant data that matches the user’s query.


## Key Modules

### `app.py`:
   - The main application logic.
   - Streamlit UI for inputting the website URL, showing scraped content, and allowing users to request specific information from the parsed content.

### `scrape.py`:
   - Contains functions to scrape the website content using Selenium, clean it using BeautifulSoup, and divide it into chunks for easier processing.

### `parse.py`:
   - Implements OpenAI GPT-based parsing using LangChain, which extracts user-specified data from the web content.
   - Handles interaction with OpenAI’s API.

## How to Run:

```bash

# Clone the repository
echo "Cloning the WebWeaver repository..."
git clone https://github.com/UppuluriKalyani/ML-Nexus

cd Generative Models/AI-Web-Scrapper/

# Install dependencies
echo "Installing required dependencies..."
pip install -r requirements.txt

# Run the Streamlit app
echo "Starting the WebWeaver app..."
streamlit run app.py
```
