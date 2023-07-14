import concurrent.futures
import io
import json
import os
import openai
import re
import random
import requests
import time
import base64
from PIL import Image
from pathlib import Path
from datetime import datetime, date, time, timezone
from dotenv import load_dotenv
from typing import List, Dict, TypedDict
from concurrent.futures import ThreadPoolExecutor, wait
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from content_main import chat_with_gpt3

#==================================================================================================
# Load Parameters
#==================================================================================================

# Load .env file
load_dotenv()

# Get the API key
openai_api_key = os.getenv("OPENAI_API_KEY", "")
# API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1-base"
# API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-0.9"
headers = {"Authorization": f"Bearer {os.getenv('STABILITY_KEY')}"}

# Use the API key
openai.api_key = openai_api_key
openai.Model.list()

# load memory directory
memory_dir = os.getenv("MEMORY_DIRECTORY", "local")
workspace_path = "./"
# The workspace_path is the path to the workspace directory.
if memory_dir == "production":
    import boto3
    workspace_path = "/tmp"
elif memory_dir == "local":
    workspace_path = "./"


class Message(TypedDict):
    role: str
    content: str
    
#==================================================================================================
# API Interaction
#==================================================================================================


def query(query_parameters: Dict[str, str]) -> bytes:
    """
     Query the VirusTotal API with the given parameters. This is a wrapper around requests. post that does not raise exceptions.
     
     @param query_parameters - A dictionary of key value pairs that are used to make the query.
     
     @return The response as a byte string or an empty string
    """
    try:
        response = requests.post(API_URL, headers=headers, json=query_parameters, timeout=120)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return b""


def stabilityai_generate(prompt: str,
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
        "size": "1280x800"
    })
    print(type(image_bytes))
    byteImgIO = io.BytesIO(image_bytes)
    image = Image.open(byteImgIO)
    directory = Path(workspace_path) / 'content'
    os.makedirs(directory, exist_ok=True)

    # Get the current timestamp and format it as a string
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    filename = f"{section}_{timestamp}.jpg"
    # Save the image object as a .jpg file with the timestamp as the filename
    image.save(directory / filename)
    print("Done")
    return filename
    
    
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


#==================================================================================================
# JSON Functions
#==================================================================================================

def sanitize_filename(filename: str) -> str:
    """
     Remove special characters from filename and replace spaces with underscores. This is useful for converting filenames to a format that can be used in a file name
     
     @param filename - The filename to clean up
     
     @return A cleaned up version of the filename ( no spaces
    """
    """Remove special characters and replace spaces with underscores in a string to use as a filename."""
    return re.sub(r'[^A-Za-z0-9]+', '_', filename)

#==================================================================================================
# URL Functions
#==================================================================================================
#==================================================================================================

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
                return s3_path
            return filename
        else:
            print("Unable to download image")
    except Exception as e:
        print(f"An error occurred while trying to download the image: {e}")
        return None

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

    image_context = chat_with_gpt3(prompt_messages, temp=0.7, p=0.8)
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
    logo_context = chat_with_gpt3(prompt_messages, temp=0.7, p=0.8)
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
