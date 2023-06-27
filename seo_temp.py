import csv
import concurrent.futures
import json
import os
import openai
import re
import random
import requests
import sys
import time
import io
import base64
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict
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
if memory_dir == "production":
    workspace_path = "./tmp"
elif memory_dir == "local":
    workspace_path = "./"


# ==================================================================================================
# API Interaction
# ==================================================================================================
 

def query(query_parameters: Dict[str, str]) -> bytes:
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
    

def generate_content_response(prompt: str,
                              temp: float,
                              p: float,
                              freq: float,
                              presence: float,
                              max_retries: int,
                              model: str) -> tuple:
    delay: float = 1  # initial delay
    exponential_base: float = 2
    jitter: bool = True
    num_retries: int = 0

    while True:
        try:
            if num_retries > max_retries:
                print(f"Max retries exceeded. The API continues to respond with an error after " + str(
                    max_retries) + " attempts.")
                return None, None, None, None  # return None if an exception was caught

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

        except openai.error.RateLimitError as e:  # rate limit error
            num_retries += 1
            print("Rate limit reached. Retry attempt " + str(num_retries + 1) + " of " + str(max_retries) + "...")
        except openai.error.Timeout as e:  # timeout error
            num_retries += 1
            print("Request timed out. Retry attempt " + str(num_retries + 1) + " of " + str(max_retries) + "...")
        except openai.error.ServiceUnavailableError:
            num_retries += 1
            print("Server Overloaded. Retry attempt " + str(num_retries + 1) + " of " + str(max_retries) + "...")
        except openai.error.InvalidRequestError as e:
            num_retries += 1
            print("Invalid Request. Retry attempt " + str(num_retries + 1) + " of " + str(max_retries) + "...")
        except openai.error.APIConnectionError as e:
            #Handle connection error here
            print(f"Failed to connect to OpenAI API: {e}Retry attempt " + str(num_retries + 1) + " of " + str(max_retries) + "...")
        except openai.error.APIError as e:
            num_retries += 1
            print(f"OpenAI API returned an API Error: {e}. Retry attempt " + str(num_retries + 1) + " of " + str(max_retries) + "...")

        # Increment the delay
        delay *= exponential_base * (1 + jitter * random.random())
        print(f"Wait for {delay} seconds.")

        time.sleep(delay)  # wait for n seconds before retrying


def generate_image_response(prompt: str,
                            size: str,
                            max_retries: int) -> str:
    delay: float = 1  # initial delay
    exponential_base: float = 2
    jitter: bool = True
    num_retries: int = 0

    while True:
        try:
            if num_retries > max_retries:
                print(f"Max retries exceeded. The API continues to respond with an error after " + str(
                    max_retries) + " attempts.")
                return ""  # return "" if an exception was caught

            print("Generating image...")
            response = openai.Image.create(
                prompt=prompt,
                n=1,
                size=size,
            )
            # print (response)
            return response['data'][0]['url']

        except openai.error.RateLimitError as e:  # rate limit error
            num_retries += 1
            print("Rate limit reached. Retry attempt " + str(num_retries + 1) + " of " + str(max_retries) + "...")
        except openai.error.Timeout as e:  # timeout error
            num_retries += 1
            print("Request timed out. Retry attempt " + str(num_retries + 1) + " of " + str(max_retries) + "...")
        except openai.error.ServiceUnavailableError:
            num_retries += 1
            print("Server Overloaded. Retry attempt " + str(num_retries + 1) + " of " + str(max_retries) + "...")
        except openai.error.InvalidRequestError as e:
            num_retries += 1
            print("Invalid Request. Retry attempt " + str(num_retries + 1) + " of " + str(max_retries) + "...")
        except openai.error.APIConnectionError as e:
            #Handle connection error here
            print(f"Failed to connect to OpenAI API: {e}Retry attempt " + str(num_retries + 1) + " of " + str(max_retries) + "...")
        except openai.error.APIError as e:
            num_retries += 1
            print(f"OpenAI API returned an API Error: {e}. Retry attempt " + str(num_retries + 1) + " of " + str(max_retries) + "...")
            
        # Increment the delay
        delay *= exponential_base * (1 + jitter * random.random())
        print(f"Wait for {delay} seconds.")
        
        time.sleep(delay)  # wait for n seconds before retrying


def chat_with_gpt3(stage: str,
                   prompt: str,
                   temp: float = 0.5,
                   p: float = 0.5,
                   freq: float = 0,
                   presence: float = 0,
                   model: str = "gpt-3.5-turbo") -> str:
    max_retries = 5
    response, prompt_tokens, completion_tokens, total_tokens = generate_content_response(prompt, temp, p, freq, presence, max_retries, model)
    if response is not None:   # If a response was successfully received
        write_to_csv((stage, prompt_tokens, completion_tokens, total_tokens, None, None))
        return response


def chat_with_dall_e(prompt: str,
                     size: str,
                     section: str) -> str:
    max_retries = 5
    url: str = generate_image_response(prompt, size, max_retries)
    if url is not None:   # If a response was successfully received
        return url

# =======================================================================================================================
# CSV Functions
# =======================================================================================================================


def write_to_csv(data: tuple):
    file_path = os.path.join(workspace_path, "token_usage.csv")
    file_exists = os.path.isfile(file_path)  # Check if file already exists
    with open(file_path, 'a+', newline='') as csvfile:
        fieldnames = ['Company Name', 'Keyword', 'Iteration', 'Stage', 'Prompt Tokens', 'Completion Tokens', 'Total Tokens', 'Price']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()  # If file doesn't exist, write the header

        csvfile.seek(0)  # Move the file pointer to the beginning of the file so we can read from the start
        last_row = None
        for last_row in csv.DictReader(csvfile):
            pass  # The loop will leave 'last_row' as the last row
        if data[0] == 'Initial':
            iteration = 0
        else:
            iteration = int(last_row['Iteration']) + 1 if last_row else 0  # If there is a last row, increment its 'Iteration' value by 1. Otherwise, start at 0
        price = 0.000002 * data[3]  # Calculate the price of the request
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
    for key, value in overrides.items():
        if isinstance(value, dict):
            # get node or create one
            node = source.setdefault(key, {})
            deep_update(node, value)
        else:
            source[key] = value
    return source

  
def processjson(jsonf: str) -> str:
    startindex = jsonf.find("{")
    endindex = jsonf.rfind("}")
    if startindex == -1 or endindex == -1:
        return ""
    else:
        jsonf = jsonf[startindex:endindex+1]
    return jsonf


def sanitize_filename(filename: str) -> str:
    """Remove special characters and replace spaces with underscores in a string to use as a filename."""
    return re.sub(r'[^A-Za-z0-9]+', '_', filename)


def url_to_base64(url: str) -> str:
    response = requests.get(url)
    if response.status_code == 200:
        # Get the content of the response
        image_data = response.content

        # Convert the image data to a base64 string
        base64_image = base64.b64encode(image_data).decode('utf-8')
        return base64_image
    else:
        print("Unable to download image")

# def fail_safe(website: str) -> str:
#     if website.find('<!DOCTYPE html>') == -1:
#         website = htmlcode
#     return website


# ##===================================================================================================
# Content Generation Methods
# ##===================================================================================================


def get_industry(topic) -> str:
    prompt = f"Generate an industry for these keywords, no explanation is needed: {topic}"
    industry = chat_with_gpt3("Industry Identification", prompt, temp=0.2, p=0.1)
    print("Industry Found")
    return industry


def get_audience(topic: str) -> List[str]:
    audienceList = []
    prompt = f"Generate a list of target audience for these keywords, no explanation is needed: {topic}"
    audience = chat_with_gpt3("Target Search", prompt, temp=0.2, p=0.1)
    audiences = audience.split('\n')  # split the keywords into a list assuming they are comma-separated
    audiences = [target.replace('"', '') for target in audiences]
    audiences = [re.sub(r'^\d+\.\s*', '', target) for target in audiences]
    audienceList.extend(audiences)
    print("Target Audience Generated")
    return audienceList


def generate_long_tail_keywords(topic: str) -> List[str]:
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
    prompt = f"Suggest 1 SEO optimized headline about '{keyword}' for the company {company_name}"
    title = chat_with_gpt3("Title Generation", prompt, temp=0.7, p=0.8)
    title = title.replace('"', '')
    print("Titles Generated")
    return title


def generate_meta_description(company_name: str,
                              topic: str,
                              keywords: str) -> str:
    print("Generating meta description...")
    prompt = f"""
    Generate a meta description for a website based on this topic: '{topic}'.
    Use these keywords in the meta description: {keywords}
    """
    meta_description = chat_with_gpt3("Meta Description Generation", prompt, temp=0.7, p=0.8)
    return meta_description


# def generate_outline(company_name: str,
#                      topic: str,
#                      industry: str,
#                      keyword: str,
#                      title: str,
#                      index: int) -> None:
#     prompt = f"""
#     Please create a comprehensive content outline for a landing page dedicated to {company_name} that centers around the topic of '{title}' and the keyword '{keyword}'. Your outline should consist of no more than seven bullet points, each preceded by a dash ("-"). Please refrain from including any concluding statements in your outline.
#     """
#     outline = chat_with_gpt3("Outline Generation", prompt, temp=0.7, p=0.8)
#     filename = f"Outline {index+1}"  # use the first keyword as the filename
#     directorypath = "outline"
#     os.makedirs(directorypath, exist_ok=True)
#     with open(os.path.join(directorypath, f'{filename}.txt'), 'w') as f:
#         f.write(outline)     


def generate_content(company_name: str,
                     topic: str,
                     industry: str,
                     keyword: str,
                     title: str) -> str:

    print("Generating Content...")
    directory_path = os.path.join(workspace_path, "content")
    os.makedirs(directory_path, exist_ok=True)
    json1 = """
    {
        "banner": {
                "h1": "...",
                "h2": "...",
                "button": [] (Pick 2 from these: Learn More, Contact Us, Get Started, Sign Up, Subscribe, Shop Now, Book Now, Get Offer, Get Quote, Get Pricing, Get Estimate, Browse Now, Try It Free, Join Now, Download Now, Get Demo, Request Demo, Request Quote, Request Appointment, Request Information, Start Free Trial, Sign Up For Free, Sign Up For Trial, Sign Up For Demo, Sign Up For Consultation, Sign Up For Quote, Sign Up For Appointment, Sign Up For Information, Sign Up For Trial, Sign Up For Demo, Sign Up For Consultation, Sign Up For Quote, Sign Up For Appointment, Sign Up For Information, Sign Up For Trial, Sign Up For Demo, Sign Up For Consultation", "Sign Up For Quote", "Sign Up For Appointment", "Sign Up For Information", "Sign Up For Trial", "Sign Up For Demo", "Sign Up For Consultation", "Sign Up For Quote", "Sign Up For Appointment", "Sign Up For Information"])
        },
        "about": {
                "h2": "About Us",
                "p": "..."
        },
        "blogs":{
            "h2": "... (Pick 1 from these: News, Articles, Insights, Resources, Customer Reviews)",
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
                },
                {
                    "h3": "...",
                    "p": "...",
                },
                {
                    "h3": "...",
                    "p": "...",
                },...
            ]
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
                       title: str) -> dict:
    description = generate_meta_description(company_name, topic, keyword)
    print(description)
    content = generate_content(company_name, topic, industry, keyword, title)
    content = processjson(content)
    contentjson = json.loads(content)
    updated_json = {"meta": {"title": title, "description": description}}
    updated_json.update(contentjson)
    print("Content Generated")
    # print(json.dumps(updated_json, indent=4))
    return updated_json


# =======================================================================================================================
# Image Generation
# =======================================================================================================================

def get_image_context(company_name: str,
                      keyword: str,
                      section: str,
                      topic: str,
                      industry: str) -> str:
    context_json = """
        {
            "description":"...",
            "size":"1024x1024"
        }
    """
    examples = """
    1) Wide shot of a sleek and modern chair design that is currently trending on Artstation, sleek and modern design, artstation trending, highly detailed, beautiful setting in the background, art by wlop, greg rutkowski, thierry doizon, charlie bowater, alphonse mucha, golden hour lighting, ultra realistic.
    2) Close-up of a modern designer handbag with beautiful background, photorealistic, unreal engine, from Vogue Magazine.
    3) Vintage-inspired watch an elegant and timeless design with intricate details, and detailed lighting, trending on Artstation, unreal engine, smooth finish, looking towards the viewer.
    4) Close-up of modern designer a minimalist and contemporary lamp design, with clean lines and detailed lighting, trending on Artstation, detailed lighting, perfect for any contemporary space.
    5) Overhead view of a sleek and futuristic concept car with aerodynamic curves, and a glossy black finish driving on a winding road with mountains in the background, sleek and stylish design, highly detailed, ultra realistic, concept art, intricate textures, interstellar background, space travel, art by alphonse mucha, greg rutkowski, ross tran, leesha hannigan, ignacio fernandez rios, kai carpenter, perfect for any casual occasion.
    6) Close-up of a designer hand-crafting a sofa with intricate details, and detailed lighting, trending on Artstation, unreal engine, smooth finish.
    7) Low angle shot of @mystyle sunglasses a modern and sleek design with reflective lenses, worn by a model standing on a city street corner with tall buildings in the background, sleek and stylish design, highly detailed, ultra realistic.
    """
    prompt = f"""
    Generate a detailed description for an image about {keyword} and {topic}. 
    i) Use these example descriptions: {examples}
    ii) Don't include any words within the picture.
    iii) Format: {context_json}
    """
    image_context = chat_with_gpt3("Image Description Generation", prompt, temp=0.7, p=0.8)
    image_cont = processjson(image_context)
    # print(image_context)
    # print(image_cont)
    imagecontext = json.loads(image_cont)
    imageurl = chat_with_dall_e(imagecontext["description"], imagecontext["size"], section)
    print(imageurl)
    image_base64 = url_to_base64(imageurl)
    return image_base64
    
    
def generate_gallery_images(company_name: str,
                            keyword: str,
                            topic: str, 
                            industry: str) -> List[str]:
    gallery = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_image_context, company_name, keyword, f"gallery {i}", topic, industry): i for i in range(8)}

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()  # Get the result of the future
                # result_base64 = url_to_base64(result)
                gallery.append(result)
            except Exception as e:
                print(f"An exception occurred during execution: {e}")
    return gallery


def image_generation(company_name: str,
                     topic: str,
                     industry: str,
                     keyword: str) -> Dict:
    print("Generating Images...")
    image_json = {
        "banner": 
            {
                "image": ""
            },
        "about": 
            {
                "image": ""
            },
        "gallery": 
            {
                "image": []
            }
    }

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Start the threads and collect the futures for non-gallery sections
       
        futures = {executor.submit(get_image_context, company_name, keyword, section, topic, industry): section for section in ["banner", "about"]}

        # Add the gallery futures
        image_json["gallery"]["image"] = (generate_gallery_images(company_name, keyword, topic, industry))

        for future in concurrent.futures.as_completed(futures):
            section = futures[future]
            try:
                image_url: list = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (section, exc))
            else:
                if image_url:
                    image_json[section]["image"] = image_url
    print("Images Generated")
    return image_json


def feature_function(company_name: str,
                     topic: str,
                     industry: str,
                     selected_keyword: str,
                     title: str) -> Dict:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        image_future = executor.submit(image_generation, company_name, topic, industry, selected_keyword)
        content_future = executor.submit(content_generation, company_name, topic, industry, selected_keyword, title)
        futures = [image_future, content_future]
        done, not_done = concurrent.futures.wait(futures, timeout=60, return_when=concurrent.futures.ALL_COMPLETED)
        try:
            image_result = image_future.result()
            content_result = content_future.result()
        except Exception as e:
            print("An exception occurred during execution: ", e)

        if image_result is None or content_result is None:
            print("Error: No results returned")
            return {}
        else:
            merged_dict = deep_update(content_result, image_result)
            return merged_dict

# =======================================================================================================================
# Main Function
# =======================================================================================================================


def main():
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

            # Generate SEO keywords
            long_tail_keywords = generate_long_tail_keywords(topic)
            for number, keyword in enumerate(long_tail_keywords):
                print(f"{number+1}. {keyword}")

            # Generate title from keyword
            selected_keyword = long_tail_keywords[random.randint(0, 4)]
            print("Selected Keyword: " + selected_keyword)
            title = generate_title(company_name, selected_keyword)
            print(title)
            
            merged_dict = feature_function(company_name, topic, industry, selected_keyword, title)
            if merged_dict is None:
                print("Error: No results returned")
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
            print(f"An exception occurred: {e}, retrying attempt {tries+1}")
            if tries < max_tries:
                tries += 1
            else:
                print(f"Maximum tries exceeded. Exiting the program.")
                flag = False
                break


if __name__ == "__main__":
    main()







# {
#     "meta": {
#         "title": "...",
#         "description": "..."
#     },
#     "logo": "...",
#     "banner": {
#             "image": "...",
#             "h1": "...",
#             "h2": "...",
#             "button": [
#                 "About Us",
#                 "Learn More"
#             ]
#     },
#     "about": {
#             "image": "...",
#             "h2": "About Us",
#             "p": "..."
#     },
#     "blogs":{
#         "h2": "Customer Review",
#         "post": [{
#                 "h3": "...",
#                 "p": "...",
#             },
#             {
#                 "h3": "...",
#                 "p": "...",
#             },
#             {
#                 "h3": "...",
#                 "p": "...",
#             }
#         ]
#     },
#     "gallery":{
#         
#     },
#     "faq":{
#         "h2": "Frequently Asked Questions",
#         "question": [{
#                 "h3": "...",
#                 "p": "...",
#             },
#             {
#                 "h3": "...",
#                 "p": "...",
#             },
#             {
#                 "h3": "...",
#                 "p": "...",
#             },...
#         ]
#     },
# }


# {
#     "meta": {
#         "title": "Experience Sustainable Luxury: Hertz's Eco-Friendly Lodges in the Heart of Malaysian Rainforest",
#         "description": "Looking for the best hotel in Malaysia? Look no further than Hertz! Our eco-friendly lodges in the Malaysian rainforest offer a unique and sustainable travel experience. Book your stay now and enjoy the natural beauty of Malaysia while minimizing your environmental impact."
#     },
#     "banner": {
#         "h1": "Experience Sustainable Luxury: Hertz's Eco-Friendly Lodges in the Heart of Malaysian Rainforest",
#         "h2": "Discover the Best Hotel in Malaysia",
#         "button": [
#             "About Us",
#             "Learn More"
#         ]
#     },
#     "about": {
#         "h2": "About Us",
#         "p": "At Hertz, we believe in providing sustainable luxury experiences for our guests. Nestled in the heart of the Malaysian rainforest, our eco-friendly lodges offer a unique opportunity to connect with nature while enjoying the utmost comfort and convenience. With a commitment to environmental conservation and responsible tourism, we strive to create a harmonious balance between luxury and sustainability."
#     },
#     "blogs": {
#         "h2": "Latest Articles",
#         "post": [
#             {
#                 "h3": "Exploring the Biodiversity of the Malaysian Rainforest",
#                 "p": "Embark on a journey through the lush greenery of the Malaysian rainforest and discover its incredible biodiversity. From rare flora and fauna to breathtaking waterfalls, our expert guides will take you on an unforgettable adventure, providing insights into the delicate ecosystem that surrounds our lodges."
#             },
#             {
#                 "h3": "The Benefits of Eco-Friendly Tourism",
#                 "p": "Learn about the positive impact of eco-friendly tourism and how it contributes to the preservation of natural resources. Discover how our lodges incorporate sustainable practices such as solar power, waste management, and local community engagement to minimize our environmental footprint."
#             },
#             {
#                 "h3": "Indulge in Luxury: The Amenities at Hertz's Eco-Friendly Lodges",
#                 "p": "Experience the ultimate in luxury at our eco-friendly lodges. From spacious rooms with panoramic rainforest views to gourmet dining options featuring locally sourced ingredients, our lodges offer an unparalleled level of comfort and relaxation. Unwind at our spa, take a dip in the infinity pool, or simply enjoy the tranquility of nature."
#             }
#         ]
#     },
#     "faq": {
#         "h2": "Frequently Asked Questions",
#         "question": [
#             {
#                 "h3": "Are the lodges fully sustainable?",
#                 "p": "Yes, our lodges are designed with sustainability in mind. We utilize renewable energy sources such as solar power, implement efficient waste management systems, and prioritize the use of eco-friendly materials in our construction and operations."
#             },
#             {
#                 "h3": "What activities are available for guests?",
#                 "p": "There are plenty of activities to choose from during your stay at our lodges. Explore the rainforest on guided nature walks, embark on thrilling wildlife safaris, or simply relax and rejuvenate in the serene surroundings. Our knowledgeable staff can assist in planning personalized itineraries based on your interests."
#             },
#             {
#                 "h3": "How can I make a reservation?",
#                 "p": "Making a reservation is easy. Simply visit our website or contact our friendly customer service team. We recommend booking in advance to secure your preferred dates as our lodges are in high demand."
#             }
#         ]
#     }
# }

# {
#     "banner": {
#         "image": "banner.jpg"},
#     "about": {
#         "image": "about.jpg"
#         },
#     "gallery": {
#         "image": ["gallery0.jpg",
#         "gallery1.jpg",
#         "gallery2.jpg",
#         "gallery3.jpg",
#         "gallery4.jpg",
#         "gallery5.jpg",
#         "gallery6.jpg",
#         "gallery7.jpg"  
#         ]
#     }
# }
        