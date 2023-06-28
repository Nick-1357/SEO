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
import pytest
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, wait
from pytest_mock import mocker
import seo_temp


# load memory directory
memory_dir = os.getenv("MEMORY_DIRECTORY", "local")
workspace_path = "./"
if memory_dir == "production":
    workspace_path = "./tmp"
elif memory_dir == "local":
    workspace_path = "./"
"""
Code Analysis

Objective:
- The objective of the function is to generate content using OpenAI's GPT-3 language model by taking a prompt as input and returning the generated content as output.

Inputs:
- prompt: a string that serves as a prompt for the GPT-3 model to generate content.
- temp: a float that controls the randomness of the generated content.
- p: a float that controls the diversity of the generated content.
- freq: a float that controls how often the model repeats itself.
- presence: a float that controls how much the model focuses on generating content related to the prompt.
- max_retries: an integer that specifies the maximum number of retries in case of errors.
- model: a string that specifies the GPT-3 model to use.

Flow:
- The function starts by setting some initial values for delay, exponential_base, jitter, and num_retries.
- It then enters a while loop that tries to generate content using the GPT-3 model.
- If the generation is successful, the function returns the generated content, prompt_tokens, completion_tokens, and total_tokens.
- If an error occurs, the function catches the error and retries the generation after a delay that increases exponentially with each retry.
- If the maximum number of retries is exceeded, the function raises an exception.

Outputs:
- response.choices[0].message['content']: a string that represents the generated content.
- response['usage']['prompt_tokens']: an integer that represents the number of tokens used for the prompt.
- response['usage']['completion_tokens']: an integer that represents the number of tokens used for the completion.
- response['usage']['total_tokens']: an integer that represents the total number of tokens used.

Additional aspects:
- The function uses the OpenAI API to generate content.
- The function handles different types of errors that may occur during the generation process, such as rate limit errors, timeout errors, and service unavailable errors.
- The function uses an exponential backoff strategy to retry the generation after an error occurs.
- The function calculates the price of the request based on the number of tokens used.
"""
class TestGenerateContentResponse:

    # Tests that the function returns None when invalid input parameters are provided
    def test_api_response_with_invalid_input(self, mocker):
        mocker.patch('openai.ChatCompletion.create', side_effect=openai.error.InvalidRequestError('Invalid input', 'param'))
        response, prompt_tokens, completion_tokens, total_tokens = seo_temp.generate_content_response('', 0.5, 0.5, 0, 0, 5, 'gpt-3.5-turbo')
        assert response is None
        assert prompt_tokens is None
        assert completion_tokens is None
        assert total_tokens is None

    # Tests that the seo_temp.generate_content_response function returns None for all output parameters when an InvalidRequestError is raised due to an invalid model name being passed to the openai.ChatCompletion.create function.
    def test_api_response_with_invalid_model_name(self, mocker):
        # Mock the openai.ChatCompletion.create function to raise an InvalidRequestError
        mocker.patch('openai.ChatCompletion.create', side_effect=openai.error.InvalidRequestError('Invalid model name', 'model'))

        # Call the seo_temp.generate_content_response function with invalid model name and a different prompt
        response, prompt_tokens, completion_tokens, total_tokens = seo_temp.generate_content_response('different prompt', 0.5, 0.5, 0, 0, 5, 'invalid model')

        # Assert that the response and tokens are None
        assert response is None
        assert prompt_tokens is None
        assert completion_tokens is None
        assert total_tokens is None

    # Tests that the function returns None when an invalid prompt is provided
    def test_api_response_with_invalid_prompt(self, mocker):
        mocker.patch('openai.ChatCompletion.create', side_effect=openai.error.InvalidRequestError('Invalid prompt', 'param'))
        response, prompt_tokens, completion_tokens, total_tokens = seo_temp.generate_content_response('', 0.5, 0.5, 0, 0, 5, 'gpt-3.5-turbo')
        assert response is None
        assert prompt_tokens is None
        assert completion_tokens is None
        assert total_tokens is None

    # Tests that the function returns None when an invalid temperature is provided
    def test_api_response_with_invalid_temperature(self, mocker):
        mocker.patch('openai.ChatCompletion.create', side_effect=openai.error.InvalidRequestError('Invalid temperature', 'temperature'))
        response, prompt_tokens, completion_tokens, total_tokens = seo_temp.generate_content_response('valid prompt', -1.0, 0.5, 0, 0, 5, 'gpt-3.5-turbo')
        assert response is None
        assert prompt_tokens is None
        assert completion_tokens is None
        assert total_tokens is None

    # Tests that the function returns None when an invalid top_p is provided
    def test_api_response_with_invalid_top_p(self, mocker):
        mocker.patch('openai.ChatCompletion.create', side_effect=openai.error.InvalidRequestError('Invalid top_p', 'top_p'))
        response, prompt_tokens, completion_tokens, total_tokens = seo_temp.generate_content_response('valid prompt', 0.5, 1.5, 0, 0, 5, 'gpt-3.5-turbo')
        assert response is None
        assert prompt_tokens is None
        assert completion_tokens is None
        assert total_tokens is None

    # Tests that the function returns None when an invalid frequency_penalty is provided
    def test_api_response_with_invalid_frequency_penalty(self, mocker):
        # Mock the openai.ChatCompletion.create method to raise error
        mocker.patch('openai.ChatCompletion.create', side_effect=openai.error.InvalidRequestError('Invalid frequency penalty', 'freq'))
        # Call the seo_temp.generate_content_response function with an invalid frequency penalty
        response = seo_temp.generate_content_response('prompt', 0.5, 0.5, -1, 0, 5, 'model')

        # Assert that the function returns None for all values in the tuple
        assert response == (None, None, None, None)

    # Tests that the function returns None when an invalid presence_penalty is provided
    def test_api_response_with_invalid_presence_penalty(self, mocker):
        mocker.patch('openai.ChatCompletion.create', side_effect=openai.error.InvalidRequestError('Invalid presence penalty', 'presence'))
        response = seo_temp.generate_content_response('prompt', 0.5, 0.5, 0, -1, 5, 'model')
        assert response == (None, None, None, None)

    # Tests that the function returns None after maximum retries are exceeded
    def test_retry_api_call_with_maximum_retries(self, mocker):
        mocker.patch('openai.ChatCompletion.create', side_effect=openai.error.APIError('Rate limit reached', {'status_code': 429}))
        response = seo_temp.generate_content_response('prompt', 0.5, 0.5, 0, 0, 2, 'model')
        assert response == (None, None, None, None)

    def test_timeout_error(self, mocker):
        mocker.patch('openai.ChatCompletion.create', side_effect=openai.error.Timeout())
        response = seo_temp.generate_content_response('Test prompt', 0.5, 0.5, 0, 0, 5, 'model')
        assert response == (None, None, None, None)
    
    def test_api_response_handles_RateLimitError(self, mocker):
        mocker.patch('openai.ChatCompletion.create', side_effect=openai.error.RateLimitError('Rate limit reached', {'status_code': 429}))
        with pytest.raises(openai.error.RateLimitError):
            response = seo_temp.generate_content_response('prompt', 0.5, 0.5, 0, 0, 2, 'model')
            assert response == (None, None, None, None)
        
    def test_api_response_handles_RateLimitError(self, mocker):
        mocker.patch('openai.ChatCompletion.create', side_effect=openai.error.ServiceUnavailableError('Rate limit reached', {'status_code': 429}))
        response = seo_temp.generate_content_response('prompt', 0.5, 0.5, 0, 0, 2, 'model')
        assert response == (None, None, None, None)
        
        # Tests that the function generates an image with valid prompt, size and section
    def test_happy_path_generate_image(self, mocker):
        class MockResponse:
            def __init__(self):
                self.content = b''
                self.status_code = 200
            
            def raise_for_status(self):
                if self.status_code != 200:
                    raise requests.exceptions.HTTPError("Error")

        class MockImage:
            def __init__(self):
                pass

            def save(self, path):
                pass

        mocker.patch('requests.post', return_value=MockResponse())
        mocker.patch('PIL.Image.open', return_value=MockImage())
        mocker.patch('os.makedirs', return_value=None)
        mocker.patch('os.path.exists', return_value=True)
        mocker.patch('os.path.isfile', return_value=True)
        mocker.patch('os.path.getsize', return_value=0)
        result = seo_temp.stabilityai_generate('test prompt', 'test size', 'test section')
        assert result == 'test section.jpg'
        assert os.path.exists('./content/test section.jpg')
        assert os.path.isfile('./content/test section.jpg')
        assert os.path.getsize('./content/test section.jpg') == 0
   
class TestWriteToCsv:
    # Tests that data is written to CSV file successfully

    # Tests that data is written to CSV file with correct fieldnames
    def test_write_to_csv_fieldnames(self):
        # Delete existing token_usage.csv file
        file_path = os.path.join(workspace_path, "token_usage.csv")
        if os.path.exists(file_path):
            os.remove(file_path)

        # Write data to CSV file
        data = ('Initial', 0, 0, 100, 'Test Company', 'Test Keyword')
        seo_temp.write_to_csv(data)

        # Check if header row matches expected fieldnames
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            header_row = next(reader)
            expected_fieldnames = ['Company Name', 'Keyword', 'Iteration', 'Stage', 'Prompt Tokens', 'Completion Tokens', 'Total Tokens', 'Price']
            assert header_row == expected_fieldnames

    # Tests that header is written to CSV file when file does not exist
    def test_write_to_csv_header(self):
        if os.path.exists(os.path.join(workspace_path, 'token_usage.csv')):
            os.remove(os.path.join(workspace_path, 'token_usage.csv'))
        data = ('Initial', 0, 0, 100, 'Test Company', 'Test Keyword')
        seo_temp.write_to_csv(data)
        with open(os.path.join(workspace_path, 'token_usage.csv'), 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert rows[0] == ['Company Name', 'Keyword', 'Iteration', 'Stage', 'Prompt Tokens', 'Completion Tokens', 'Total Tokens', 'Price']

    # Tests that data is not written to CSV file with invalid data tuple
    def test_write_to_csv_invalid_data(self):
        data = ('Initial', 0, 0, 0, None, None)
        seo_temp.write_to_csv(data)
        with open(os.path.join(workspace_path, 'token_usage.csv'), 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert rows[-1] != ['Initial', '0', '0', '0', 'None', 'None', '0', '0.0']

class TestDeepUpdate:
    # Tests that seo_temp.deep_update returns an empty dictionary when both source and overrides are empty dictionaries
    def test_empty_dicts(self):
        source = {}
        overrides = {}
        result = seo_temp.deep_update(source, overrides)
        assert result == {}

    # Tests that seo_temp.deep_update returns a dictionary with the same key-value pairs as overrides when overrides is a shallow copy of source
    def test_shallow_copy(self):
        source = {'a': 1, 'b': 2}
        overrides = {'a': 3, 'b': 4}
        result = seo_temp.deep_update(source, overrides)
        assert result == {'a': 3, 'b': 4}

    # Tests that seo_temp.deep_update returns a dictionary with new key-value pairs added from overrides when overrides has new keys not present in source
    def test_new_keys(self):
        source = {'a': 1, 'b': 2}
        overrides = {'c': 3, 'd': 4}
        result = seo_temp.deep_update(source, overrides)
        assert result == {'a': 1, 'b': 2, 'c': 3, 'd': 4}

    # Tests that seo_temp.deep_update returns a dictionary with key-value pairs updated from overrides when overrides has keys that are already present in source
    def test_existing_keys(self):
        source = {'a': 1, 'b': 2}
        overrides = {'a': 3, 'b': 4}
        result = seo_temp.deep_update(source, overrides)
        assert result == {'a': 3, 'b': 4}

    # Tests that seo_temp.deep_update returns a dictionary with nested dictionaries updated correctly from overrides when overrides has nested dictionaries with keys already present in source
    def test_nested_dicts(self):
        source = {'a': {'b': 1, 'c': 2}}
        overrides = {'a': {'b': 3, 'd': 4}}
        result = seo_temp.deep_update(source, overrides)
        assert result == {'a': {'b': 3, 'c': 2, 'd': 4}}

    # Tests that seo_temp.deep_update returns a dictionary with None values updated correctly from overrides when source and overrides have nested dictionaries with None values
    def test_none_values(self):
        source = {'a': {'b': None, 'c': None}}
        overrides = {'a': {'b': 1, 'd': 2}}
        result = seo_temp.deep_update(source, overrides)
        assert result == {'a': {'b': 1, 'c': None, 'd': 2}}

    # Tests that seo_temp.deep_update returns an empty dictionary when source is None
    def test_behaviour(self):
        source = {'a': 1, 'b': {'c': 2}}
        overrides = {}
        result = seo_temp.deep_update(source, overrides)
        assert result == {'a': 1, 'b': {'c': 2}}

    # Tests that the function returns the source dictionary when overrides is None
    def test_behaviour(self):
        source = {'a': 1, 'b': {'c': 2}}
        overrides = None
        result = seo_temp.deep_update(source, overrides)
        assert result == source

    # Tests that seo_temp.deep_update raises a TypeError when overrides is not a dictionary
    def test_non_dict_overrides(self):
        source = {'a': 1}
        overrides = 'not a dictionary'
        assert seo_temp.deep_update(source, overrides) == source

    # Tests that seo_temp.deep_update raises a TypeError when source is not a dictionary
    def test_source_not_dict(self):
        with pytest.raises(TypeError):
            overrides = {'a': 1}
            seo_temp.deep_update(1, overrides)