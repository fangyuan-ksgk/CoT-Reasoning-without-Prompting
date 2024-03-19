import requests
from bs4 import BeautifulSoup

def convert_webpage_to_txt(url):
    # Send a GET request to the webpage
    response = requests.get(url)

    # Create a BeautifulSoup object to parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract the text content from the parsed HTML
    text = soup.get_text()

    # Remove leading and trailing whitespace
    text = text.strip()

    # Replace multiple whitespace characters with a single space
    text = ' '.join(text.split())

    # Generate the output file name
    output_file = 'webpage_content.txt'

    # Save the text content to a file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(text)

    print(f"Webpage content saved as {output_file}")

# Prompt the user for the web link
url = input("Enter the web link: ")

# Call the function to convert the webpage to a text file
convert_webpage_to_txt(url)