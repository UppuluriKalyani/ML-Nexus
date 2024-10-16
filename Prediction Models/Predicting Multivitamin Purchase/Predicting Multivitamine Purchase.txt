from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# Initialize the Chrome driver
driver = webdriver.Chrome()
driver.get("https://www.1mg.com/")

# Close the city update modal if it exists
try:
    update_modal = driver.find_element(By.XPATH, "//div[@class='UpdateCityModal_update-btn_2qmN1 UpdateCityModalbtn__oMW5n']")
    update_modal.click()
    time.sleep(10)
except:
    pass

# Wait for the page to load
time.sleep(2)

# Search for multivitamins
search_bar = driver.find_element(By.XPATH, "//input[@id='srchBarShwInfo']")
search_bar.send_keys("Multivitamins")

# Click the search button
search_button = driver.find_element(By.XPATH, "//div[@class='header_search_icon']")
search_button.click()

# Wait for the search results to load
time.sleep(5)

while True:
    # Extract and print product details
    product_boxes = driver.find_elements(By.XPATH, "//div[@class='style__product-box___3oEU6']")
    for product_box in product_boxes:
        print("--------------------------")
        try:
            # Open the product in a new tab
            product_url = product_box.find_element(By.XPATH, ".//a").get_attribute("href")
            driver.execute_script("window.open(arguments[0], '_blank');", product_url)
            time.sleep(2)

            # Switch to the newly opened tab
            driver.switch_to.window(driver.window_handles[1])

            # Gather information
            try:
                brand_name = driver.find_element(By.XPATH, "//h1").text
                print("Brand:", brand_name)
            except:
                print("Brand: Not available")

            try:
                cost = driver.find_element(By.XPATH, "//span[@class='PriceBoxPlanOption__offer-price___3v9x8 PriceBoxPlanOption__offer-price-cp___2QPU_']").text
                print("Cost:", cost)
            except:
                print("Cost: Not available")

            try:
                rating = driver.find_element(By.XPATH, "//span[@class='RatingDisplay__ratings-header___ZNj5b']").text
                print("Rating:", rating)
            except:
                print("Rating: Not available")

            try:
                tablet_count = driver.find_element(By.XPATH, "//span[@class='PackSizeLabel__single-packsize___3KEr_']").text
                print("No of tablets:", tablet_count)
            except:
                print("No of tablets: Not available")

            try:
                no_of_users = driver.find_element(By.XPATH, "//span[@class='SocialCue__views-text___1CTJI']").text
                print("No of users:", no_of_users)
            except:
                print("No of users: Not available")

            try:
                product_highlights = driver.find_element(By.XPATH, "//div[@class='ProductHighlights__highlights-text___dc-WQ']").text
                print("Product highlights:\n", product_highlights)
                highlights = product_highlights.split('\n')  # Assuming each highlight is on a new line
                highlights = [highlight for highlight in highlights if highlight.strip()]  # Filter out any empty strings
                print("Number of product highlights: ", len(highlights))
            except:
                print("Product highlights: Not available")

            try:
                product_description = driver.find_element(By.XPATH, "//div[@class='ProductDescription__product-description___1PfGf']").text
                print("Product description:\n", product_description)

                # Extract key ingredients section and count the key ingredients
                key_ingredients_text = product_description.split("Key Ingredients:")[1]
                key_ingredients_section = key_ingredients_text.split("Key Benefits:")[0]
                key_ingredients_list = key_ingredients_section.split(',')
                key_ingredients_count = len([item for item in key_ingredients_list if item.strip()])
                print("Number of key ingredients: ", key_ingredients_count)
            except:
                print("Product description: Not available")

            # Close the tab
            driver.close()

            # Switch back to the main window
            driver.switch_to.window(driver.window_handles[0])

        except Exception as e:
            print("Exception occurred:", e)

    # Check for the "Next" button and navigate to the next page if available
    try:
        next_button = driver.find_element(By.XPATH, "//span[@class='style__next___2Cubq']")
        driver.execute_script("arguments[0].click();", next_button)
        time.sleep(5)  # Wait for the next page to load
    except:
        print("No more pages.")
        break

# Close the browser
driver.quit()
