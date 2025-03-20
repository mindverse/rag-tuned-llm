import openai
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import json
import os
from tqdm import tqdm
import re
import pandas as pd

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_API_BASE_URL")
MODEL = os.getenv("OPENAI_API_MODEL")

def load_parquet(file_path):
    """
    Load a parquet file.
    """
    return pd.read_parquet(file_path)

def format_context(chunks):
    """
    Format the context information.
    """
    context = ""
    for idx, chunk in enumerate(chunks):
        context += f"Context: \n{chunk}\n\n"
    return context

def save_to_json(list_to_load, file_name):
    """
    Save the generated questions to a JSON file.
    """
    with open(file_name, 'w') as json_file:
        json.dump(list_to_load, json_file, indent=4, ensure_ascii=False)

    print(f"Questions saved to {file_name}")

def multi_process_request(all_messages, max_workers, func, structure=None):
    with ThreadPoolExecutor(max_workers=min(max_workers, len(all_messages))) as executor:
        futures = [(i, executor.submit(func, messages, structure)) if structure is not None else (i, executor.submit(func, messages)) for i, messages in enumerate(all_messages)]
        results = [None] * len(all_messages) 
        for i, future in tqdm(futures):
            try:
                result = future.result()
                results[i] = result
            except Exception as e:
                results[i] = f"Raise ERROR: {e} WHEN GENERATE RESPONSE"

    return results

def process_request(messages, format_class=None):
    try:
        client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
        model = MODEL
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None

def process_request_structered(messages, format_class):
    try:
        client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
        model = MODEL
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=format_class,
        )
        message = completion.choices[0].message
        if message.parsed:
            return message.parsed
        else:
            return message.refusal
    except Exception as e:
        return f"Error occurred: {str(e)}"

def remove_surrogates(string):
    return string.encode('utf-16', 'surrogatepass').decode('utf-16', 'ignore')

def clean_non_utf8_chars(string):
    return string.encode("utf-8", "ignore").decode("utf-8", "ignore")

def remove_unicode_escapes(text):
    return re.sub(r'\\u[0-9A-Fa-f]{4}', '', text)
