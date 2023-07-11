import csv
import concurrent.futures
import io
import json
import os

import boto3
import openai
import re
import random
import requests
import sys
import time
import base64
from PIL import Image
from pathlib import Path
from datetime import datetime, date, time, timezone
from dotenv import load_dotenv
from typing import List, Dict, TypedDict
from concurrent.futures import ThreadPoolExecutor, wait
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

# Load .env file
load_dotenv()

# Get the API key
openai_api_key = os.getenv("OPENAI_API_KEY", "")
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1-base"
headers = {"Authorization": f"Bearer {os.getenv('STABILITY_KEY')}"}

# Use the API key
openai.api_key = openai_api_key
openai.Model.list()

# load memory directory
memory_dir = os.getenv("MEMORY_DIRECTORY", "local")
workspace_path = "./"
# The workspace_path is the path to the workspace directory.
if memory_dir == "production":
    workspace_path = "/tmp"
elif memory_dir == "local":
    workspace_path = "./"


class Message(TypedDict):
    role: str
    content: str

# ==================================================================================================
# API Interaction
# ==================================================================================================


def query(query_parameters: Dict[str, str]) -> bytes:
    """
     Query the VirusTotal API with the given parameters. This is a wrapper around requests. post that does not raise exceptions.
     
     @param query_parameters - A dictionary of key value pairs that are used to make the query.
     
     @return The response as a byte string or an empty string
    """
    try:
        response = requests.post(API_URL, headers=headers, json=query_parameters, timeout=10)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return b""


def stabilityai_generate(prompt: str,
                         size: str,
                         section: str) -> str:
    """
    Generate stabilityai jpg image. This is a wrapper around query that allows you to specify the size and section of the image you want to generate
    
    @param prompt - prompt to provide to the user
    @param size - size of the image in pixels ( must be between 1 and 1024 )
    @param section - section of the image that will be generated ( ex : images. jpg)
    
    @return path to generated jpg
    """
    print(f"Generating {section} image...")
    image_bytes = query({
        "inputs": f"{prompt}",
        "size": size
    })
    byteImgIO = io.BytesIO(image_bytes)
    image = Image.open(byteImgIO)
    directory = Path(workspace_path) / 'content'
    os.makedirs(directory, exist_ok=True)
    image.save(directory / f'{section}.jpg')
    print("Done")
    return f'{section}.jpg'
    

def generate_content_response(prompt: str | List[Message],
                              temp: float,
                              p: float,
                              freq: float,
                              presence: float,
                              max_retries: int,
                              model: str) -> tuple:
    """
    Generate a response to a content request. This is a wrapper around openai. ChatCompletion.
    
    @param prompt - The prompt to respond to
    @param temp - The temperatures to use for the response
    @param p - The p - value to use for the response
    @param freq - The frequency penalty 
    @param presence - The presence penalty
    @param max_retries - The maximum number of retries before terminating
    @param model - The model to use for the API call.
    
    @return A tuple containing the response ( if any )
    """
    delay: float = 1  # initial delay
    exponential_base: float = 2
    jitter: bool = True
    num_retries: int = 0

    # This function is used to wait for a chat completion.
    while True:
        if num_retries >= max_retries:
            print(f"Max retries exceeded. The API continues to respond with an error after " + str(
                max_retries) + " attempts.")
            return None, None, None, None  # return None if an exception was caught
        else:
            try:
                if isinstance(prompt, str):
                    response = openai.ChatCompletion.create(
                        model=f"{model}",
                        messages=[
                                {"role": "system", "content": "You are an web designer with the objective to identify search engine optimized long-tail keywords and generate contents, with the goal of generating website contents and enhance website's visibility, driving organic traffic, and improving online business performance."},
                                {"role": "user", "content": prompt}
                            ],
                        temperature=temp,
                        # max_tokens=2500,
                        top_p=p,
                        frequency_penalty=freq,
                        presence_penalty=presence,
                    )
                    # print (response)
                    return response.choices[0].message['content'], response['usage']['prompt_tokens'], response['usage']['completion_tokens'], response['usage']['total_tokens']
                elif isinstance(prompt, List):
                    # print("Prompt: ", prompt)
                    response = openai.ChatCompletion.create(
                        model=f"{model}",
                        messages=prompt,
                        temperature=temp,
                        # max_tokens=2500,
                        top_p=p,
                        frequency_penalty=freq,
                        presence_penalty=presence,
                    )
                    # print (response)
                    return response.choices[0].message['content'], response['usage']['prompt_tokens'], response['usage']['completion_tokens'], response['usage']['total_tokens']

            except openai.error.RateLimitError as e:  # rate limit error
                num_retries += 1
                print("Rate limit reached. Retry attempt " + str(num_retries) + " of " + str(max_retries) + "...")
            except openai.error.Timeout as e:  # timeout error
                num_retries += 1
                print("Request timed out. Retry attempt " + str(num_retries) + " of " + str(max_retries) + "...")
            except openai.error.ServiceUnavailableError:
                num_retries += 1
                print("Server Overloaded. Retry attempt " + str(num_retries) + " of " + str(max_retries) + "...")
            except openai.error.InvalidRequestError as e:
                num_retries += 1
                print("Invalid Chat Request. Retry attempt " + str(num_retries) + " of " + str(max_retries) + "...")
            except openai.error.APIConnectionError as e:
                #Handle connection error here
                print(f"Failed to connect to OpenAI API: {e}Retry attempt " + str(num_retries) + " of " + str(max_retries) + "...")
            except openai.error.APIError as e:
                num_retries += 1
                print(f"OpenAI API returned an API Error: {e}. Retry attempt " + str(num_retries) + " of " + str(max_retries) + "...")

            # Increment the delay
            delay *= exponential_base * (1 + jitter * random.random())
            print(f"Wait for {round(delay, 2)} seconds.")

        time.sleep(delay)  # wait for n seconds before retrying


def chat_with_gpt3(stage: str,
                   prompt: str | List[Message],
                   temp: float = 0.5,
                   p: float = 0.5,
                   freq: float = 0,
                   presence: float = 0,
                   model: str = "gpt-3.5-turbo") -> str:
    """
    Generate a response to a prompt with GPT
    
    @param stage - The stage of the t2w
    @param prompt - The prompt to send to the user ( s )
    @param temp - Randomeness ( default 0.5 )
    @param p - Randomeness ( default 0.5 )
    @param freq - Frequency Penalty ( default 0 )
    @param presence - Presence Penalty ( default 0 )
    @param model - Model to use ( default gpt - 3. 5 - turbo )
    
    @return The response or None if something went wrong ( in which case the user should be prompted
    """
    max_retries = 5       
    response, prompt_tokens, completion_tokens, total_tokens = generate_content_response(prompt, temp, p, freq, presence, max_retries, model)
    # If a response was received return the response.
    if response is not None:   # If a response was successfully received
        write_to_csv((stage, prompt_tokens, completion_tokens, total_tokens, None, None))
        return response
    else:
        return None
    

def generate_image_response(prompt: str,
                            max_retries: int) -> str:
    """
        Generates and returns image URL. This is a wrapper around openai. Image. create that handles rate limiting and timeouts
        
        @param prompt - prompt to send to the DALL E API
        @param max_retries - maximum number of retries to make before terminating
        
        @return url of the generated image or " " if an error
    """
    delay: float = 1  # initial delay
    exponential_base: float = 2
    jitter: bool = True
    num_retries: int = 0

    # This function is used to generate an image and return the image.
    while True:
        if num_retries >= max_retries:
            print(f"Max retries exceeded. The API continues to respond with an error after " + str(
                max_retries) + " attempts.")
            return ""  # return "" if an exception was caught
        else:
            try:
                print("Generating image...")
                response = openai.Image.create(
                    prompt=prompt,
                    n=1,
                    size="1024x1024",
                )
                # print (response)
                return response['data'][0]['url']

            except openai.error.RateLimitError as e:  # rate limit error
                num_retries += 1
                print("Rate limit reached. Retry attempt " + str(num_retries) + " of " + str(max_retries) + "...")
            except openai.error.Timeout as e:  # timeout error
                num_retries += 1
                print("Request timed out. Retry attempt " + str(num_retries) + " of " + str(max_retries) + "...")
            except openai.error.ServiceUnavailableError:
                num_retries += 1
                print("Server Overloaded. Retry attempt " + str(num_retries) + " of " + str(max_retries) + "...")
            except openai.error.InvalidRequestError as e:
                num_retries += 1
                print("Invalid Image Request. Retry attempt " + str(num_retries) + " of " + str(max_retries) + "...")
                # print("Prompt: ", prompt)
            except openai.error.APIConnectionError as e:
                num_retries += 1
                print(f"Failed to connect to OpenAI API: {e}Retry attempt " + str(num_retries) + " of " + str(max_retries) + "...")
            except openai.error.APIError as e:
                num_retries += 1
                print(f"OpenAI API returned an API Error: {e}. Retry attempt " + str(num_retries) + " of " + str(max_retries) + "...")
                
            # Increment the delay
            delay *= exponential_base * (1 + jitter * random.random())
            print(f"Wait for {round(delay, 2)} seconds.")
            
            time.sleep(delay)  # wait for n seconds before retrying


def chat_with_dall_e(prompt: str,
                     section: str) -> str:
    """
    Prompt the user for a response. This is a dalle version of : func : ` chat_with_dall `

    @param prompt - The prompt for image
    @param section - The section of the image that is being generated

    @return The URL of the response or None if no response was
    """
    max_retries = 3
    url: str = generate_image_response(prompt, max_retries)
    # Returns the URL of the response.
    if url is not None:   # If a response was successfully received
        return url
    else:
        return None

# =======================================================================================================================
# CSV Functions
# =======================================================================================================================


def write_to_csv(data: tuple):
    """
     Writes the data to csv file. This is a function that takes a tuple of data and writes it to the token_usage. csv file
     
     @param data - tuple of data to write into the csv file
     
    """
    file_path = os.path.join(workspace_path, "token_usage.csv")
    file_exists = os.path.isfile(file_path)  # Check if file already exists
    with open(file_path, 'a+', newline='') as csvfile:
        fieldnames = ['Company Name', 'Keyword', 'Iteration', 'Stage', 'Prompt Tokens', 'Completion Tokens', 'Total Tokens', 'Price']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # Write header to file if file exists
        if not file_exists:
            writer.writeheader()  # If file doesn't exist, write the header

        csvfile.seek(0)  # Move the file pointer to the beginning of the file so we can read from the start
        last_row = None
        # This function will read the last row of the csv file
        for last_row in csv.DictReader(csvfile):
            pass  # The loop will leave 'last_row' as the last row
        # The initialize iteration number
        if data[0] == 'Initial':
            iteration = 0
        else:
            iteration = int(last_row['Iteration']) + 1 if last_row else 0  # If there is a last row, increment its 'Iteration' value by 1. Otherwise, start at 0
        price = 0.000004 * data[3]  # Calculate the price of the request
        writer.writerow({'Company Name': data[4], 'Keyword': data[5], 'Iteration': iteration, 'Stage': data[0], 'Prompt Tokens': data[1], 'Completion Tokens': data[2], 'Total Tokens': data[3], 'Price': float(price)})

    # file_exists = os.path.isfile('token_usage.csv')  # Check if file already exists
    # with open('token_usage.csv', 'a', newline='') as csvfile:
    #     fieldnames = ['Company Name', 'Keyword', 'Iteration', 'Stage', 'Prompt Tokens', 'Completion Tokens', 'Total Tokens', 'Price']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     if not file_exists:
    #         writer.writeheader()
    #     writer.writerow({'Company Name': company_name, 'Keyword': topic, 'Iteration': 0, 'Stage': 'Initial', 'Prompt Tokens': 0, 'Completion Tokens': 0, 'Total Tokens': 0, 'Price': 0})

    
# ##==================================================================================================
# JSON Functions
# ##==================================================================================================

def deep_update(source, overrides):
    """
     Update a dict with values from another dict. This is a deep update function for the config.
     
     @param source - The dict to update. It will be modified in place
     @param overrides - The dict to override the values in
     
     @return The source dict with the values
    """
    # Return the source code if overrides is not None.
    if not overrides or not isinstance(overrides, dict):
        return source
    # Updates the source dictionary with the given overrides.
    for key, value in overrides.items():
        # Set the value of the source node.
        if isinstance(value, dict):
            # get node or create one
            node = source.setdefault(key, {})
            deep_update(node, value)
        else:
            source[key] = value
    return source


def update_json(data1):
    """
     Updates the JSON for front-end
     
     @param data1 - The JSON to update
     
     @return The updated JSON as a Python dictionary
    """
    # convert the JSON strings to Python dictionaries:
    data2 = {
        "layouts": [
            {
                "layout": "Layout_header_1",
                "value": {
                    "style": [],
                    "image": "",
                    "position": 0
                }
            },
            {
                "layout": "Layout_centered_image_1",
                "value": {
                    "style": [],
                    "position": 1,
                    "button": [
                        {
                            "name": "",
                            "layout": 1,
                            "style": []
                        }
                    ],
                    "image": "",
                    "h1": {
                        "value": "",
                        "html": "same as value",
                        "style": []
                    },
                    "h2": {
                        "value": "",
                        "html": "same as value",
                        "style": []
                    }
                }
            },
            {
                "layout": "Layout_right_image_1",
                "value": {
                    "style": [],
                    "position": 2,
                    "h2": {
                        "value": "",
                        "html": "same as value",
                        "style": []
                    },
                    "paragraph": {
                        "value": "",
                        "html": "same as value",
                        "style": []
                    },
                    "image": ""
                }
            },
            {
                "layout": "Layout_three_blogs_1",
                "value": {
                    "style": [],
                    "position": 3,
                    "h2": {
                        "value": "",
                        "html": "same as value",
                        "style": []
                    },
                    "blogs": [
                        {
                            "h2": {
                                "value": "",
                                "html": "same as value",
                                "style": []
                            },
                            "paragraph": {
                                "value": "",
                                "html": "same as value",
                                "style": []
                            }
                        }
                    ]
                }
            },
            {
                "layout": "Layout_contact_us_1",
                "value": {
                    "style": [],
                    "position": 4,
                    "h1": {
                        "value": "Have a Question?",
                        "html": "Have a Question?",
                        "style": []
                    },
                    "h4": {
                        "value": "Contact us today!",
                        "html": "Contact us today!",
                        "style": []
                    },
                    "image": ""
                }
            },
            {
                "layout": "Layout_frequently_asked_questions_1",
                "value": {
                    "style": [],
                    "position": 5,
                    "h2": {
                        "value": "",
                        "html": "same as value",
                        "style": []
                    },
                    "Faq": [
                        {
                            "h3": {
                                "value": "",
                                "html": "same as value",
                                "style": []
                            }
                        }
                    ]
                }
            },
            {
                "layout": "Layout_gallery_1",
                "value": {
                    "style": [],
                    "h2": {
                        "value": "Gallery",
                        "html": "Gallery",
                        "style": []
                    },
                    "position": 6,
                    "images": [
                        {
                            "url": "",
                            "alt": ""
                        }
                    ]
                }
            },
            {
                "layout": "Layout_right_image_2",
                "value": {
                    "style": [],
                    "position": 7,
                    "h2": {
                        "value": "",
                        "html": "same as value",
                        "style": []
                    },
                    "paragraph": {
                        "value": "",
                        "html": "same as value",
                        "style": []
                    },
                    "image": ""
                }
            },
            {
                "layout": "Layout_map_1",
                "value": {
                    "style": [],
                    "position": 8,
                    "h2": {
                        "value": "Map",
                        "html": "same as value",
                        "style": []
                    },
                    "map_src": ""
                }
            },
            {
                "layout": "Layout_footer_1",
                "value": {
                    "style": [],
                    "position": 9,
                    "h1": {
                        "value": "Contact Info",
                        "html": "Contact Info",
                        "style": []
                    },
                    "paragraph": [
                        {
                            "value": "",
                            "html": "same as value",
                            "style": []
                        }
                    ],
                    "image": ""
                }
            }
        ],
        "meta_data": {
            "title": "",
            "description": ""
        }
    }
    
    # update the second JSON data with the data from the first JSON:
    data2['layouts'][0]['value']['image'] = data1['logo']['image']
    
    data2['layouts'][1]['value']['h1']['value'] = data1['banner']['h1']
    data2['layouts'][1]['value']['h1']['html'] = data1['banner']['h1']
    data2['layouts'][1]['value']['h2']['value'] = data1['banner']['h2']
    data2['layouts'][1]['value']['h2']['html'] = data1['banner']['h2']
    data2['layouts'][1]['value']['button'] = data1['banner']['button']
    data2['layouts'][1]['value']['image'] = data1['banner']['image']

    data2['layouts'][2]['value']['h2']['value'] = data1['about']['h2']
    data2['layouts'][2]['value']['h2']['html'] = data1['about']['h2']
    data2['layouts'][2]['value']['paragraph']['value'] = data1['about']['p']
    data2['layouts'][2]['value']['paragraph']['html'] = data1['about']['p']
    data2['layouts'][2]['value']['image'] = data1['about']['image']

    data2['layouts'][3]['value']['h2']['value'] = data1['blogs']['h2']
    data2['layouts'][3]['value']['h2']['html'] = data1['blogs']['h2']
    data2['layouts'][3]['value']['blogs'] =[{'h2': {'value': post['h3'], 'html': post['h3'], 'style': []}, 'paragraph': {'value': post['p'], 'html': post['p'], 'style': []}} for post in data1['blogs']['post']]

    data2["layouts"][4]['value']['image'] = data1['contactus']['image']
    
    data2['layouts'][5]['value']['h2']['value'] = data1['faq']['h2']
    data2['layouts'][5]['value']['h2']['html'] = data1['faq']['h2']
    data2['layouts'][5]['value']['Faq'] = [{'h3': {'value': q['h3'], 'html': q['h3']}, 'paragraph': {'value': q['p'], 'html': q['p']}} for q in data1['faq']['question']]

    data2["layouts"][7]['value']['h2']['html'] = data1['blog2']['h2']
    data2["layouts"][7]['value']['h2']['value'] = data1['blog2']['h2']
    data2["layouts"][7]['value']['paragraph']['value'] = data1['blog2']['p']
    data2["layouts"][7]['value']['paragraph']['html'] = data1['blog2']['p']
    data2["layouts"][7]['value']['image'] = data1['blog2']['image']
       
    data2['layouts'][6]['value']['images'] = [{'url': img, 'alt': ''} for img in data1['gallery']['image']]

    data2['layouts'][8]['value']['map_src'] = data1['map']['map_src']
    
    data2['layouts'][9]['value']['paragraph'] = [{'value': para, 'html': para} for para in data1['footer']['info']]
    data2['layouts'][9]['value']['image'] = data1['logo']['image']
    
    data2['meta_data']['title'] = data1['meta']['title']
    data2['meta_data']['description'] = data1['meta']['description']
    # convert the updated data back to a JSON string:
    updated_json = json.dumps(data2)
    return data2


def processjson(jsonf: str) -> str:
    """
     Processes a JSON string and returns the result. If the JSON cannot be parsed an empty string is returned
     
     @param jsonf - the JSON string to process
     
     @return the JSON string or an empty string if there was
    """
    startindex = jsonf.find("{")
    endindex = jsonf.rfind("}")
    if startindex == -1 or endindex == -1:
        return ""
    else:
        try:
            json.loads(jsonf[startindex:endindex+1])
            return jsonf[startindex:endindex+1]
        except ValueError:
            return ""


def sanitize_filename(filename: str) -> str:
    """
     Remove special characters from filename and replace spaces with underscores. This is useful for converting filenames to a format that can be used in a file name
     
     @param filename - The filename to clean up
     
     @return A cleaned up version of the filename ( no spaces
    """
    """Remove special characters and replace spaces with underscores in a string to use as a filename."""
    return re.sub(r'[^A-Za-z0-9]+', '_', filename)


def sanitize_location(location: str) -> str:
    """
     Sanitizes location to prevent XSS.
     
     @param location - The location to sanitize.
     
     @return The sanitized location as a string in the format " %20 " or " %2C "
    """
    url_safe_address = location.replace(" ", "%20")
    url_safe_address = url_safe_address.replace(",", "%2C")
    return url_safe_address


def url_to_base64(url: str) -> str:
    """
     Download an image from a URL and convert it to a base64 string.
     
     @param url - The URL of the image to download. It should be a URL that points to an image.
     
     @return The base64 string of the image or None if there was an error
    """
    try:
        response = requests.get(url)
        # Returns the image data as a base64 encoded string
        if response.status_code == 200:
            # Get the content of the response
            image_data = response.content

            # Convert the image data to a base64 string
            base64_image = base64.b64encode(image_data).decode('utf-8')
            return base64_image
        else:
            print("Unable to download image")
    except Exception as e:
        print(f"An error occurred while trying to download the image: {e}")
        return None


def url_to_jpg(url: str, section: str) -> str:
    """
     Downloads and saves the image to jpg. This is used to generate the image for the user
     
     @param url - The url of the image
     @param section - The section of the image to be downloaded
     
     @return The filename of the image or None if there was an error
    """
    try:
        response = requests.get(url)
        if response.status_code == 200:
            image_data = response.content
            
            byteImgIO = io.BytesIO(image_data)
            image = Image.open(byteImgIO)
            directory = Path(workspace_path) / 'content'
            os.makedirs(directory, exist_ok=True)

            # Get the current timestamp and format it as a string
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            filename = f"{section}_{timestamp}.jpg"
            # Save the image object as a .jpg file with the timestamp as the filename
            image.save(directory / filename)

            if memory_dir == "production":
                campaign_id = os.getenv("CAMPAIGN_ID", "0")
                bucket_name = os.getenv("BUCKET_NAME", None)

                s3_path = str(campaign_id) + "/asset/" + filename
                s3 = boto3.client('s3')
                print("Uploading {}...".format(s3_path))
                s3.upload_file(Filename=directory / filename,
                               Bucket=bucket_name,
                               Key=s3_path)
            return filename
        else:
            print("Unable to download image")
    except Exception as e:
        print(f"An error occurred while trying to download the image: {e}")
        return None


# ##===================================================================================================
# Content Generation Methods
# ##===================================================================================================


def get_industry(topic: str) -> str:
    """
     Get industry for keywords.
     
     @param topic - keyword from user input
     
     @return identified industry for keyword
    """
    prompt = f"Generate an industry for these keywords, no explanation is needed: {topic}"
    industry = chat_with_gpt3("Industry Identification", prompt, temp=0.2, p=0.1)
    print("Industry Found")
    return industry


def get_audience(topic: str) -> List[str]:
    """
     Get a list of audiences for a topic.
     
     @param topic - The topic for which to get the audience.
     
     @return A list of target audiences for the topic. It is empty if user cancels
    """
    audienceList = []
    prompt = f"Generate a list of target audience for these keywords, no explanation is needed: {topic}"
    audience = chat_with_gpt3("Target Search", prompt, temp=0.2, p=0.1)
    audiences = audience.split('\n')  # split the keywords into a list assuming they are comma-separated
    audiences = [target.replace('"', '') for target in audiences]
    audiences = [re.sub(r'^\d+\.\s*', '', target) for target in audiences]
    audienceList.extend(audiences)
    print("Target Audience Generated")
    return audienceList


def get_location(topic: str) -> str:
    """
     Generate location from user keyword.
     @param topic - topic of the address.
     @return a string of the form " street / city / postcode/ state / country
    """
    print("Identifying Location..")
    prompt = f"Generate an address (Building number, Street name, Postal Code, City/Town name, State, Country) in one line for this keywords, no explanation is needed: {topic}"
    location = chat_with_gpt3("Location Identification", prompt, temp=0.2, p=0.1)
    print("Location Found")
    return location


def generate_long_tail_keywords(topic: str) -> List[str]:
    """
     Generate 5 SEO optimised long tail keywords related to the topic.
     
     @param topic - topic to generate long tail keywords for
     
     @return list of keywords for the topic as a list of string
    """
    keyword_clusters = []
    prompt = f"Generate 5 SEO-optimized long-tail keywords related to the topic: {topic}."
    keywords_str = chat_with_gpt3("Keyword Clusters Search", prompt, temp=0.2, p=0.1)
    keywords = keywords_str.split('\n')  # split the keywords into a list assuming they are comma-separated
    keywords = [keyword.replace('"', '') for keyword in keywords]
    keywords = [re.sub(r'^\d+\.\s*', '', keyword) for keyword in keywords]
    keyword_clusters.extend(keywords)
    print("Keywords Generated")
    return keyword_clusters


def generate_title(company_name: str,
                   keyword: str) -> str:
    """
    Generate and return title for a given companies headline.

    @param company_name - The name of the company
    @param keyword - The keyword for the title to be generated.

    @return The title as a string
    """
    prompt = f"Suggest 1 SEO optimized headline about '{keyword}' for the company {company_name}"
    title = chat_with_gpt3("Title Generation", prompt, temp=0.7, p=0.8)
    title = title.replace('"', '')
    print("Titles Generated")
    return title


def generate_meta_description(company_name: str,
                              topic: str,
                              keywords: str) -> str:
    """
    Generate a meta description for a website based on a topic and keywords.
    
    @param company_name - Company name to be used in the message
    @param topic - Topic for which we want to generate a meta description
    @param keywords - Keywords that will be used in the meta description
    
    @return Meta description as a string
    """
    print("Generating meta description...")
    prompt = f"""
    Generate a meta description for a website based on this topic: '{topic}'.
    Use these keywords in the meta description: {keywords}
    """
    meta_description = chat_with_gpt3("Meta Description Generation", prompt, temp=0.7, p=0.8)
    return meta_description


def generate_footer(company_name: str,
                    topic: str,
                    industry: str,
                    keyword: str,
                    title: str,
                    location: str) -> dict:
    """
     Generate a footer. We need to generate an email to the Google Maps site and the map's url so it can be embedded in the template
     
     @param company_name - The company name of the user
     @param location - The location of the user in the google maps
     
     @return The JSON representation of the template's footer
    """
    print("Generating footer")
    start = random.choice(["+601", "+603"])
    rest = "".join(random.choice("0123456789") for _ in range(8))  # we generate 8 more digits since we already have 2
    number = start + rest
    email = "info@" + company_name.lower().replace(" ", "") + ".com"
    address = location.replace("1. ", "", 1)
    url_location = sanitize_location(address)
    mapurl = f"https://maps.google.com/maps?q={url_location}&t=&z=10&ie=UTF8&iwloc=&output=embed"
    
    footer_json = {
        "map": {
            "map_src": ""
        },
        "footer": {
            "info": []
        }
    }
    footer_json['map']['map_src'] = mapurl
    footer_json['footer']['info'].extend([number, email, address])
    return footer_json


def generate_content(company_name: str,
                     topic: str,
                     industry: str,
                     keyword: str,
                     title: str,
                     location: str) -> str:
    """
    Generates content for the template. This is a function that takes care of the creation of the content
    
    @param company_name - The name of the company
    @param topic - The keyword of the users
    @param industry - The industry of the topic
    @param keyword - The keyword found
    @param title - The title of the content
    
    @return The JSON string of the content
    """

    print("Generating Content...")
    directory_path = os.path.join(workspace_path, "content")
    os.makedirs(directory_path, exist_ok=True)
    json1 = """
    {
        "banner": {
                "h1": "...",
                "h2": "...",
                "button": [
                    {
                        "name": "...", 
                        "layout": 1
                        "style": []
                    },
                    {
                        "name": "...",
                        "layout": 2
                        "style": []
                    }...
                ] (Pick from these: Learn More, Contact Us, Get Started, Sign Up, Subscribe, Shop Now, Book Now, Get Offer, Get Quote, Get Pricing, Get Estimate, Browse Now, Try It Free, Join Now, Download Now, Get Demo, Request Demo, Request Quote, Request Appointment, Request Information, Start Free Trial, Sign Up For Free, Sign Up For Trial, Sign Up For Demo, Sign Up For Consultation, Sign Up For Quote, Sign Up For Appointment, Sign Up For Information, Sign Up For Trial, Sign Up For Demo, Sign Up For Consultation, Sign Up For Quote, Sign Up For Appointment, Sign Up For Information, Sign Up For Trial, Sign Up For Demo, Sign Up For Consultation, Sign Up For Quote, Sign Up For Appointment, Sign Up For Information, Sign Up For Trial, Sign Up For Demo, Sign Up For Consultation,  Sign Up For Quote, Sign Up For Appointment, Sign Up For Information)
        },
        "about": {
                "h2": "About Us",
                "p": "..."
        },
        "blogs":{
            "h2": "... (e.g.: Our Services, Customer Reviews, Insights, Resources)",
            "post": [{
                    "h3": "...",
                    "p": "...",
                },
                {
                    "h3": "...",
                    "p": "...",
                },
                {
                    "h3": "...",
                    "p": "...",
                }
            ]
        },
        "faq":{
            "h2": "Frequently Asked Questions",
            "question": [{
                    "id": 1,
                    "h3": "...",
                    "p": "...",
                },
                {
                    "id": 2,
                    "h3": "...",
                    "p": "...",
                },
                {
                    "id": 3,
                    "h3": "...",
                    "p": "...",
                },
                {
                    "id": 4,
                    "h3": "...",
                    "p": "...",
                },
                {
                    "id": 5,
                    "h3": "...",
                    "p": "...",
                },...
            ]
        },
        "blog2": {
                "h2": "Our Mission",
                "p": "..."
        }
    }
    """
    prompt = f"""
    Create a SEO optimized website content with the following specifications:
    Company Name: {company_name}
    Title: {title}
    Industry: {industry}
    Core Keywords: {topic}
    Keywords: {keyword}
    Format: {json1}
    Requirements:
    1) Make sure the content length is 700 words.
    2) The content should be engaging and unique.
    3) The FAQ section should follow the SERP and rich result guidelines
    """
    content = chat_with_gpt3("Content Generation", prompt, temp=0.7, p=0.8, model="gpt-3.5-turbo-16k")
    return content


def content_generation(company_name: str,
                       topic: str,
                       industry: str,
                       keyword: str,
                       title: str,
                       location: str) -> dict:
    """
    Generates and returns content. This is the main function of the content generation process
    
    @param company_name - The name of the company
    @param topic - The topic of the industry to generate
    @param industry - The industry of the industry to generate
    @param keyword - The keyword of the industry to generate
    @param title - The title of the industry to generate
    @param location - The location of the industry to generate
    
    @return dict with meta information about the content
    """
    print("Starting Content Process")
    try:
        description = generate_meta_description(company_name, topic, keyword)
        content = generate_content(company_name, topic, industry, keyword, title, location)
        footer = generate_footer(company_name, topic, industry, keyword, title, location)
    except Exception as e:
        return {'error': str(e)}
    content = processjson(content)
    contentjson = json.loads(content)
    updated_json = {"meta": {"title": title, "description": description}}
    updated_json.update(contentjson)
    updated_json.update(footer)
    print("Content Generated")
    # print(json.dumps(updated_json, indent=4))
    return updated_json


# =======================================================================================================================
# Image Generation
# =======================================================================================================================

def get_image(company_name: str,
              keyword: str,
              section: str,
              topic: str,
              industry: str) -> str:
    """
    Generate a context for an image. It is used to determine the location of the image and the context of the industry
    
    @param company_name - The name of the company
    @param keyword - The keyword that is being viewed in the context
    @param section - The section that is being viewed in the context
    @param topic - The topic that is being viewed in the context
    @param industry - The industry that is being viewed in the context
    
    @return The context of the industry as a string
    """
    print("Generating Context...")
    examples = """
    Wide shot of a sleek and modern chair design that is currently trending on Artstation, sleek and modern design, artstation trending, highly detailed, beautiful setting in the background, art by wlop, greg rutkowski, thierry doizon, charlie bowater, alphonse mucha, golden hour lighting, ultra realistic./
    Close-up of a modern designer handbag with beautiful background, photorealistic, unreal engine, from Vogue Magazine./
    Vintage-inspired watch an elegant and timeless design with intricate details, and detailed lighting, trending on Artstation, unreal engine, smooth finish, looking towards the viewer./
    Close-up of modern designer a minimalist and contemporary lamp design, with clean lines and detailed lighting, trending on Artstation, detailed lighting, perfect for any contemporary space./
    Overhead view of a sleek and futuristic concept car with aerodynamic curves, and a glossy black finish driving on a winding road with mountains in the background, sleek and stylish design, highly detailed, ultra realistic, concept art, intricate textures, interstellar background, space travel, art by alphonse mucha, greg rutkowski, ross tran, leesha hannigan, ignacio fernandez rios, kai carpenter, perfect for any casual occasion./
    Close-up of a designer hand-crafting a sofa with intricate details, and detailed lighting, trending on Artstation, unreal engine, smooth finish./
    Low angle shot of a modern and sleek design with reflective lenses, worn by a model standing on a city street corner with tall buildings in the background, sleek and stylish design, highly detailed, ultra realistic./
    Saw and sawdust, blurred workshop background, 3D, digital art./
    Easy bake oven, fisher-price, toy, bright colors, blurred playroom background, natural-lighting./
    Fine acoustic guitar, side angle, natural lighting, bioluminescence./
    Tained glass window of fish, side angle, rubble, dramatic-lighting, light rays, digital art./
    Photo angles – a macro shot of the subject, or an overhead shot or a drone shot./
    Lighting – studio lighting, indoor lighting, outdoor lighting, backlit shots. /
    Photo lens effects – fisheye lens, double exposure /
    """
    prompt = f"""
    Generate 1 short paragraph about the detailed description of an image about {keyword}.
    The image should also be about {topic} 
    Use these as example descriptions: {examples}
    """

    prompt_messages: List[Message] = [
        {"role": "system",
         "content": "You are an web designer with the objective to create a stunning, unique and attractive design for the company to gain more traffic on the company's website."},
        {"role": "user",
         "content": "Generate 1 short paragraph about the detailed description of an image about wood cutting carpentry workshop. The image should also be about carpentry workshop."},
        {"role": "assistant",
         "content": "Saw and sawdust, blurred workshop background, 3D, digital art."},
        {"role": "user",
         "content": "Generate 1 short paragraph about the detailed description of an image about affordable toy oven for children. The image should also be about toy oven."},
        {"role": "assistant",
         "content": "Easy bake oven, fisher-price, toy, bright colors, blurred playroom background, natural-lighting."},
        {"role": "user",
         "content": "Generate 1 short paragraph about the detailed description of an image about top acoustic guitar brands for professionals. The image should also be about acoustic guitar."},
        {"role": "assistant",
         "content": "Fine acoustic guitar, side angle, natural lighting, bioluminescence."},
        {"role": "user",
         "content": "Generate 1 short paragraph about the detailed description of an image about Fish aquarium digital art gallery. The image should also be about fish aquarium digital art."},
        {"role": "assistant",
         "content": "Tained glass window of fish, side angle, rubble, dramatic-lighting, light rays, digital art."},
        {"role": "user",
         "content": "Generate 1 short paragraph about the detailed description of an image about Contemporary ergonomic chair design. The image should also be about modern chair."},
        {"role": "assistant",
         "content": "Wide shot of a sleek and modern chair design that is currently trending on Artstation, sleek and modern design, artstation trending, highly detailed, beautiful setting in the background, art by wlop, greg rutkowski, thierry doizon, charlie bowater, alphonse mucha, golden hour lighting, ultra realistic."},
        {"role": "user",
         "content": "Generate 1 short paragraph about the detailed description of an image about Trendy modern designer handbags for women. The image should also be about modern designer handbag."},
        {"role": "assistant",
         "content": "Close-up of a modern designer handbag with beautiful background, photorealistic, unreal engine, from Vogue Magazine."},
        {"role": "user",
         "content": "Generate 1 short paragraph about the detailed description of an image about Luxury vintage-inspired and timeless watch. The image should also be about vintage-inspired timeless design watch."},
        {"role": "assistant",
         "content": "Vintage-inspired watch an elegant and timeless design with intricate details, and detailed lighting, trending on Artstation, unreal engine, smooth finish, looking towards the viewer."},
        {"role": "user",
         "content": "Generate 1 short paragraph about the detailed description of an image about best modern designers lamp design. The image should also be about electrical lightings store."},
        {"role": "assistant",
         "content": "Close-up of modern designer a minimalist and contemporary lamp design, with clean lines and detailed lighting, trending on Artstation, detailed lighting, perfect for any contemporary space."},
        {"role": "user",
         "content": "Generate 1 short paragraph about the detailed description of an image about award winning artistic design for a futuristic concept car. The image should also be about futuristic concept car."},
        {"role": "assistant",
         "content": "Overhead view of a sleek and futuristic concept car with aerodynamic curves, and a glossy black finish driving on a winding road with mountains in the background, sleek and stylish design, highly detailed, ultra realistic, concept art, intricate textures, interstellar background, space travel, art by alphonse mucha, greg rutkowski, ross tran, leesha hannigan, ignacio fernandez rios, kai carpenter, perfect for any casual occasion."},
        {"role": "user",
         "content": "Generate 1 short paragraph about the detailed description of an image about finest hand-crafted quality sofa. The image should also be about sofa manufacturer."},
        {"role": "assistant",
         "content": "Close-up of a designer hand-crafting a sofa with intricate details, and detailed lighting, trending on Artstation, unreal engine, smooth finish."},
        {"role": "user",
         "content": "Generate 1 short paragraph about the detailed description of an image about Trendy designer sunglasses for summer. The image should also be about sunglasses."},
        {"role": "assistant",
         "content": "Low angle shot of a modern and sleek design with reflective lenses, worn by a model standing on a city street corner with tall buildings in the background, sleek and stylish design, highly detailed, ultra realistic."},
        {"role": "user",
         "content": f"Generate 1 short paragraph about the detailed description of an image about {keyword}. The image should also be about {topic} "}
    ]

    image_context = chat_with_gpt3("Image Description Generation", prompt_messages, temp=0.7, p=0.8)
    # print(image_context)
    image_context += "Detailed 4K photorealistic. No fonts or text."
    imageurl = chat_with_dall_e(image_context, section)
    print(imageurl)
    image_jpg = url_to_jpg(imageurl, section)
    # image_base64 = url_to_base64(imageurl)
    return image_jpg


def generate_logo(company_name: str,
                  keyword: str,
                  section: str,
                  topic: str,
                  industry: str) -> str:
    """
    Generate a logo for a company. This is a function that can be used to generate a logo for an industry that provides a topic and keyword
    
    @param company_name - The name of the company
    @param topic - The topic to generate a logo for
    @param keyword - The keyword to generate a logo for
    @param industry - The industry for which we want to generate a logo
    
    @return The path to the generated logo or None if none
    """
    
    print("Generating Logo")
    prompt = f"""
    Describe the details and design of a logo for the company that provides {topic} in the {industry} industry.
    Only talk about the objects and the color.
    Example: 
    "Flat vector logo of a curved wave, blue, trending on Dribble"/
    "Line art logo of a owl, golden, minimal, solid black background"/
    "Gradient color logo, a gradient in 2 circles"/
    "Geometrical logo of a pyramid, dreamy pastel color palette, gradient color"
    "Organic logo, shape of a leaf"/
    "Typographical logo, floral, letter” A”, serif typeface"/
    "Emblem of chess team, royal, coat of arms, golden color, knight"
    "hamburger 3D logo, very cute shape, miniature small scale painting style, minimalism, lite object style, up view, matte, white background, soft round form, ultra high definition details, 8k"/
    "Minimal logo of a cafe, a coffee bean, gradient brown color"/
    "Graphic logo of a red wine company, eagle, modern, classy, high end, red and gold — v 5"
    "A 2d, symmetrical, flat logo for a blockchain company that is sleek and simple. It should be of black shade and should be subtle."/ 
    Write it in a few sentences.
    """
    
    prompt_messages: List[Message] = [
        {"role": "system",
         "content": "You are an web designer with the objective to create a stunning and unique logo to attract the attention of people."},
        {"role": "user",
         "content": "Describe the details and design of a logo for the company that provides pet services in Malaysia in the Pet industry"},
        {"role": "assistant",
         "content": "Line art logo of a owl, golden, minimal, solid black background"},
        {"role": "user",
         "content": "Describe the details and design of a logo for the company that provides insurance in the insurance industry"},
        {"role": "assistant",
         "content": "Geometrical logo of a pyramid, dreamy pastel color palette, gradient color."},
        {"role": "user",
         "content": "Describe the details and design of a logo for the company that provides printing in the Printing industry"},
        {"role": "assistant",
         "content": "Typographical logo, floral, letter” A”, serif typeface"},
        {"role": "user",
         "content": "Describe the details and design of a logo for the company that provides E-Sports in Malaysia in the Video Game industry"},
        {"role": "assistant",
         "content": "Emblem of chess team, royal, coat of arms, golden color, knight"},
        {"role": "user",
         "content": "Describe the details and design of a logo for the company that provides Flower Delivery Services in the Floral industry"},
        {"role": "assistant",
         "content": "Elegant and feminine logo for a florist, pastel color, minimal."},
        {"role": "user",
         "content": "Describe the details and design of a logo for the company that provides Best Burger in Malaysia in the Food and Beverage industry"},
        {"role": "assistant",
         "content": "hamburger 3D logo, very cute shape, miniature small scale painting style, minimalism, lite object style, up view, matte, white background, soft round form, ultra high definition details, 8k."},
        {"role": "user",
         "content": "Describe the details and design of a logo for the company that provides Best Cafe in Malaysia in the Food and Beverage industry"},
        {"role": "assistant",
         "content": "Minimal logo of a cafe, a coffee bean, gradient brown color."},
        {"role": "user",
         "content": "Describe the details and design of a logo for the company that provides Best Wine Company in Malaysia in the Wine industry"},
        {"role": "assistant",
         "content": "Graphic logo of a red wine company, eagle, modern, classy, high end, red and gold."},
        {"role": "user",
         "content": "Describe the details and design of a logo for the company that provides Music Services in the Music industry"},
        {"role": "assistant",
         "content": "boho style logo design, sun and wave"},
        {"role": "user",
         "content": "Describe the details and design of a logo for the company that provides Best Bar in Malaysia in the Hospitality and Entertainment industry"},
        {"role": "assistant",
         "content": "Outline logo of a bar, a glass of cocktail, flat design, neon light, dark background"},
        {"role": "user",
         "content": "Describe the details and design of a logo for the company that provides Recycling Services in Malaysia in the Waste Management and Recycling industry"},
        {"role": "assistant",
         "content": "Globe logo, green and blue, glossy base, 3d rendering, white background, isometric, translucent, technology sense, studio light, C4D, blender, clean, hyper-detailed"},
        {"role": "user",
         "content": "Describe the details and design of a logo for the company that provides Best Headphone Company in Malaysia in the Audio industry"},
        {"role": "assistant",
         "content": "Logo of a music company, headphone, splashing, futuristic, cyberpunk "},
        {"role": "user",
         "content": "Describe the details and design of a logo for the company that provides Best Sushi Resturant in Malaysia in the Food and Beverage industry"},
        {"role": "assistant",
         "content": "Japanese style logo of a sushi restaurant, a sashimi bowl with blue waves — ar 1:1 — niji 5"},
        {"role": "user",
         "content": "Describe the details and design of a logo for the company that provides Game Design Services in Malaysia in the Software industry"},
        {"role": "assistant",
         "content": "Mascot for a video game company, fox, japanese style — ar 1:1 — niji 5 — style cute"},
        {"role": "user",
         "content": "Describe the details and design of a logo for the company that provides Best Surfing Course in Malaysia in the Sports and Recreation industry"},
        {"role": "assistant",
         "content": "Flat vector logo of a curved wave, blue, trending on Dribble"},
        {"role": "user",
         "content": f"Describe the details and design of a logo for the companythat provides {topic} in the {industry} industry."}
    ]
    logo_context = chat_with_gpt3("Logo Description Generation", prompt_messages, temp=0.7, p=0.8)
    logo_context += " with no text. No fonts included."
    print(logo_context)
    # logo_context = "The newest f1 car but perodua brand"
    imageurl = chat_with_dall_e(logo_context, "Logo")
    print(imageurl)
    image_jpg = url_to_jpg(imageurl, section="logo")
    # image_base = url_to_base64(imageurl)
    return image_jpg
    
    
def generate_gallery_images(company_name: str,
                            keyword: str,
                            section: str,
                            topic: str, 
                            industry: str) -> List[str]:
    """
        Generate gallery images for a company. This is a thread safe function to call get_image in parallel
        
        @param company_name - The company's name
        @param keyword - The generated keyword
        @param topic - User's keyword
        @param industry - The industry of the topic
        
        @return A list of image ids that were generated from DALL E 
    """
    gallery = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_image, company_name, keyword, f"gallery{i}", topic, industry): i for i in range(8)}

        # Get the result of all futures in concurrent. futures. as_completed.
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()  # Get the result of the future
                gallery.append(result)
            except Exception as e:
                print(f"An exception occurred during execution: {e}")
    return gallery


def image_generation(company_name: str,
                     topic: str,
                     industry: str,
                     keyword: str) -> Dict:
    """
    Generates images for a topic industry and keyword. This function is used to generate a json file that can be uploaded to Snapchat
    
    @param company_name - The name of the company
    @param topic - User's keyword
    @param industry - The industry of topic
    @param keyword - The keyword that will be used for the image generation
    
    @return A dict with the name of image for each entry
    """
    print("Starting Image Process...")
    image_json = {
        "logo": {
            "image": ""
        },
        "banner": 
            {
                "image": ""
            },
        "about": 
            {
                "image": ""
            },
        "contactus":
            {
                "image": ""
            },
        "blog2":
            {
                "image": ""
            },
        "gallery": 
            {
                "image": []
            }
        
    }
    image_json["logo"]["image"] = generate_logo(company_name, keyword, "Logo", topic, industry)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Start the threads and collect the futures for non-gallery sections
       
        futures = {executor.submit(get_image, company_name, keyword, section, topic, industry): section for section in ["banner", "about", "contactus", "blog2"]}

        # Add the gallery futures

        # Returns the image url of the image.
        for future in concurrent.futures.as_completed(futures):
            section = futures[future]
            try:
                image: str = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (section, exc))
            else:
                # Set image_url to the image_json section
                if image:
                    image_json[section]["image"] = image
                    
    image_json["gallery"]["image"] = (generate_gallery_images(company_name, keyword, "gallery", topic, industry))            
        
    print("Images Generated")
    return image_json


def feature_function(company_name: str,
                     topic: str,
                     industry: str,
                     selected_keyword: str,
                     title: str,
                     location: str) -> Dict:
    """
    This function takes as input the values to be used in the feature function.
    
    @param company_name - The name of the company
    @param topic - User's keyword
    @param industry - The industry of the feature
    @param selected_keyword - Randomly selected keyword
    @param title - The generated title
    @param location - The generated location 
    
    @return A dictionary with the result of the content and image generation function or empty
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        image_future = executor.submit(image_generation, company_name, topic, industry, selected_keyword)
        content_future = executor.submit(content_generation, company_name, topic, industry, selected_keyword, title, location)
        futures = [image_future, content_future]
        done, not_done = concurrent.futures.wait(futures, timeout=60, return_when=concurrent.futures.ALL_COMPLETED)
        try:
            image_result = image_future.result()
            content_result = content_future.result()
        except Exception as e:
            print("An exception occurred during execution: ", e)

        # Update the result of the image and content.
        if image_result is None or content_result is None:
            print("Error: No results returned")
            return {}
        else:
            merged_dict = deep_update(content_result, image_result)
            # print(json.dumps(merged_dict, indent=4))
            final_result = update_json(merged_dict)
            # print(json.dumps(final_result, indent=4))
            return final_result

# =======================================================================================================================
# Main Function
# =======================================================================================================================


def main():
    """
     Main function to get data from the user. Args : None
    """
    # Get the company name and topic from the user
    flag = True
    tries = 0
    max_tries = 2
    try:
        company_name = sys.argv[1]
        topic = sys.argv[2]
    except IndexError:
        company_name = input("Company Name: ")
        topic = input("Your Keywords: ")
    
    while flag:
        try:
            # Open token.csv to track token usage
            write_to_csv(("Initial", 0, 0, 0, company_name, topic))

            # Generate industry 
            industry = get_industry(topic)
            print(industry)
            
            location = get_location(topic)
            print(location)

            # Generate SEO keywords
            long_tail_keywords = generate_long_tail_keywords(topic)
            for number, keyword in enumerate(long_tail_keywords):
                print(f"{number+1}. {keyword}")

            # Generate title from keyword
            selected_keyword = long_tail_keywords[random.randint(0, 4)]
            print("Selected Keyword: " + selected_keyword)
            title = generate_title(company_name, selected_keyword)
            print(title)
            
            merged_dict = feature_function(company_name, topic, industry, selected_keyword, title, location)
            # Write the merged_dict to a data.json file.
            if merged_dict is None:
                print("Error: No results returned")
                # If the maximum number of tries exceeded the program exits.
                if tries < max_tries:
                    tries += 1
                else:
                    print(f"Maximum tries exceeded. Exiting the program.")
                    flag = False
                    break
            else:
                flag = False
                # Write to JSON file
                directory_path = os.path.join(workspace_path, "content")
                os.makedirs(directory_path, exist_ok=True)
                with open(os.path.join(directory_path, f'data.json'), 'w', encoding='utf-8') as f:
                    json.dump(merged_dict, f, ensure_ascii=False, indent=4)
                
                # End procedures
                write_to_csv(("Complete", 0, 0, 0, company_name, topic))
                
        except Exception as e:
            tries += 1
            print(f"An exception occurred: {e}, retrying attempt {tries}")
            # If the maximum number of tries exceeded print out the program.
            if tries <= max_tries:
                continue
            else:
                print(f"Maximum tries exceeded. Exiting the program.")
                break


# main function for the main module
if __name__ == "__main__":
    main()
