import selenium.webdriver as wd
from selenium.webdriver.chrome.service import Service
import time
from bs4 import BeautifulSoup

def scrape_webiste(website):
    print("Strated ...")
    
    chrome_path = "chromedriver-win64\chromedriver-win64\chromedriver.exe"
    options = wd.ChromeOptions()
    driver = wd.Chrome(service= Service(chrome_path) , options= options)
     
    try :
         driver.get(website)
         print("Page Loaded ... ")
         html = driver.page_source
     #     time.sleep(1)
         return html
    finally:
         driver.quit()

def get_html_body(website):
     soup = BeautifulSoup(website , "html.parser")
     body_content = soup.body
     
     if body_content:
          return str(body_content)
     else:
          return ""

def clean_body(website):
     soup = BeautifulSoup(website , "html.parser")
     
     for script_style in soup(["script" , "style"]):
          script_style.extract()
     
     clean_text = soup.get_text(separator="\n")
     clean_text = "\n".join(line.strip() for line in clean_text.splitlines() if line.strip())
     
     return clean_text
     
def make_batches(clean_text , max_len = 5000):
     return [
          clean_text[i: i+max_len] for i in range(0 , len(clean_text) , max_len )
     ]
     
