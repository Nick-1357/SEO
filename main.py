import concurrent.futures
import json
import os
import random
import sys
from typing import List, Dict, TypedDict

from .content_main import get_industry, get_location, generate_long_tail_keywords, generate_title, content_generation, processjson
from .image_main import image_generation
from .utils import language


memory_dir = os.getenv("MEMORY_DIRECTORY", "local")
workspace_path = "./"
# The workspace_path is the path to the workspace directory.
if memory_dir == "production":
    workspace_path = "/tmp"
elif memory_dir == "local":
    workspace_path = "./"

language_state = language.en


def load_language_state():

    global language_state

    lang = os.getenv("LANGUAGE")

    language_state = language.language_locale.get(lang, language.en)
    print("language loaded: ", lang, "\n", language_state.contact_us_today)

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
                    "description": "header",
                    "style": [],
                    "images": [
                        {
                            "file_name": "",
                            "alt": ""
                        }
                    ],
                    "position": 0
                }
            },
            {
                "layout": "Layout_centered_image_1",
                "value": {
                    "description": "banner",
                    "style": [],
                    "position": 1,
                    "button": [
                        {
                            "name": "",
                            "layout": 2,
                            "style": []
                        }
                    ],
                    "images": [
                        {
                            "file_name": "",
                            "alt": ""
                        }
                    ],
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
                    "description": "about",
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
                    "images": [
                        {
                            "file_name": "",
                            "alt": ""
                        }
                    ]
                }
            },
            {
                "layout": "Layout_three_blogs_1",
                "value": {
                    "description": "blogs",
                    "style": [],
                    "position": 3,
                    "h2": {
                        "value": "",
                        "html": "same as value",
                        "style": []
                    },
                    "blogs": [
                        {
                            "h3": {
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
                    "description": "contact",
                    "style": [],
                    "position": 4,
                    "h2": {
                        "value": "Have a Question?",
                        "html": "Have a Question?",
                        "style": []
                    },
                    "paragraph": {
                        "value": "Contact us today!",
                        "html": "Contact us today!",
                        "style": []
                    },
                    "images": [
                        {
                            "file_name": "",
                            "alt": ""
                        }
                    ],
                    "button": [
                        {
                            "name": "Submit",
                            "style": []
                        }
                    ],
                }
            },
            {
                "layout": "Layout_frequently_asked_questions_1",
                "value": {
                    "description": "faq",
                    "style": [],
                    "position": 5,
                    "h2": {
                        "value": "",
                        "html": "same as value",
                        "style": []
                    },
                    "faq": [
                        {
                            "h3": {
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
                "layout": "Layout_gallery_1",
                "value": {
                    "description": "gallery",
                    "style": [],
                    "h2": {
                        "value": "Gallery",
                        "html": "Gallery",
                        "style": []
                    },
                    "position": 6,
                    "images": [
                        {
                            "file_name": "",
                            "alt": ""
                        }
                    ]
                }
            },
            {
                "layout": "Layout_right_image_1",
                "value": {
                    "description": "mission",
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
                    "images": [
                        {
                            "file_name": "",
                            "alt": ""
                        }
                    ]
                }
            },
            {
                "layout": "Layout_map_1",
                "value": {
                    "description": "map",
                    "style": [],
                    "position": 8,
                    "h2": {
                        "value": "Map",
                        "html": "Map",
                        "style": []
                    },
                    "map_src": ""
                }
            },
            {
                "layout": "Layout_footer_1",
                "value": {
                    "description": "footer",
                    "style": [],
                    "position": 9,
                    "h2": {
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
                    "images": [
                        {
                            "file_name": "",
                            "alt": ""
                        }
                    ]
                }
            }
        ],
        "meta_data": {
            "title": "",
            "description": ""
        }
    }
    
    # update the second JSON data with the data from the first JSON:
    data2['layouts'][0]['value']['images']: list = [
        {
            "file_name": data1['logo']['image'],
            "alt": ""
        }
    ]

    # Layout_centered_image_1
    data2['layouts'][1]['value']['h1']['value'] = data1['banner']['h1']
    data2['layouts'][1]['value']['h1']['html'] = data1['banner']['h1']
    data2['layouts'][1]['value']['h2']['value'] = data1['banner']['h2']
    data2['layouts'][1]['value']['h2']['html'] = data1['banner']['h2']
    data2['layouts'][1]['value']['button'] = data1['banner']['button']
    data2['layouts'][1]['value']['images']: list = [
        {
            "file_name": data1['banner']['image'],
            "alt": ""
        }
    ]

    # Layout_right_image_1
    data2['layouts'][2]['value']['h2']['value'] = data1['about']['h2']
    data2['layouts'][2]['value']['h2']['html'] = data1['about']['h2']
    data2['layouts'][2]['value']['paragraph']['value'] = data1['about']['p']
    data2['layouts'][2]['value']['paragraph']['html'] = data1['about']['p']
    data2['layouts'][2]['value']['images']: list = [
        {
            "file_name": data1['about']['image'],
            "alt": ""
        }
    ]

    # Layout_three_blogs_1
    data2['layouts'][3]['value']['h2']['value'] = data1['blogs']['h2']
    data2['layouts'][3]['value']['h2']['html'] = data1['blogs']['h2']
    data2['layouts'][3]['value']['blogs'] = [{'h3': {'value': post['h3'], 'html': post['h3'], 'style': []}, 'paragraph': {'value': post['p'], 'html': post['p'], 'style': []}} for post in data1['blogs']['post']]

    # Layout_contact_us_1
    data2['layouts'][4]['value']['h2']['value'] = language_state.have_a_question
    data2['layouts'][4]['value']['h2']['html'] = language_state.have_a_question
    data2['layouts'][4]['value']['paragraph']['value'] = language_state.contact_us_today
    data2['layouts'][4]['value']['paragraph']['html'] = language_state.contact_us_today
    data2['layouts'][4]['value']['button'][0]['name'] = language_state.submit
    data2["layouts"][4]['value']['images']: list = [
        {
            "file_name": data1['contactus']['image'],
            "alt": ""
        }
    ]

    # Layout_frequently_asked_questions_1
    data2['layouts'][5]['value']['h2']['value'] = data1['faq']['h2']
    data2['layouts'][5]['value']['h2']['html'] = data1['faq']['h2']
    data2['layouts'][5]['value']['faq'] = [{'h3': {'value': q['h3'], 'html': q['h3'], 'style': []}, 'paragraph': {'value': q['p'], 'html': q['p'], 'style': []}} for q in data1['faq']['question']]

    # Layout_gallery_1
    data2['layouts'][6]['value']['h2']['value'] = language_state.gallery
    data2['layouts'][6]['value']['h2']['html'] = language_state.gallery
    data2['layouts'][6]['value']['images'] = [{'file_name': img, 'alt': ''} for img in data1['gallery']['image']]

    # Layout_right_image_1
    data2["layouts"][7]['value']['h2']['html'] = language_state.mission
    data2["layouts"][7]['value']['h2']['value'] = language_state.mission
    data2["layouts"][7]['value']['paragraph']['value'] = data1['mission']['p']
    data2["layouts"][7]['value']['paragraph']['html'] = data1['mission']['p']
    data2["layouts"][7]['value']['images']: list = [
        {
            "file_name": data1['mission']['image'],
            "alt": ""
        }
    ]

    # Layout_map_1
    data2["layouts"][8]['value']['h2']['html'] = language_state.map
    data2["layouts"][8]['value']['h2']['value'] = language_state.map
    data2['layouts'][8]['value']['map_src'] = data1['map']['map_src']

    # Layout_footer_1
    data2["layouts"][9]['value']['h2']['html'] = language_state.contact_info
    data2["layouts"][9]['value']['h2']['value'] = language_state.contact_info
    data2['layouts'][9]['value']['paragraph'] = [{'value': para, 'html': para, 'style': []} for para in data1['footer']['info']]
    data2['layouts'][9]['value']['images']: list = [
        {
            "file_name": data1['logo']['image'],
            "alt": ""
        }
    ]

    # meta_data
    data2['meta_data']['title'] = data1['meta']['title']
    data2['meta_data']['description'] = data1['meta']['description']
    # convert the updated data back to a JSON string:
    updated_json = json.dumps(data2)
    return data2

# ==================================================================================================
# JSON Generating Function
# ==================================================================================================


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
        image_future = executor.submit(image_generation, topic, industry, selected_keyword)
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
