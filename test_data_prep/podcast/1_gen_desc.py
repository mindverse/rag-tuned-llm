import openai
import json
import re
import os

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_API_BASE_URL")
MODEL = os.getenv("OPENAI_API_MODEL")

def process_request(messages):
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

def remove_emojis(string):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F" # emoticons
        u"\U0001F300-\U0001F5FF" # symbols & pictographs
        u"\U0001F680-\U0001F6FF" # transport & map symbols
        u"\U0001F1E0-\U0001F1FF" # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", 
        flags=re.UNICODE
    )
    
    return emoji_pattern.sub(r'', string)

def remove_unicode_escapes(text):
    return re.sub(r'\\u[0-9A-Fa-f]{4}', '', text)

long_text_dir = "./podcast_data.txt"
SYS = """You are an professional agent to summarize a long text document. Organize the summary logically, with clear sections corresponding to the different episode types. Focus on the summarization of each type rather than specific details or episode.
"""
USR = """Here is a long text document that needs to be summarized:

{long_text}

Please generate a comprehensive summary of the document according to the provided guidelines.
"""
with open(long_text_dir, 'r', encoding='utf-8') as f:
    long_text = f.read()
    long_text = remove_unicode_escapes(long_text)
    print(f"Text length: {len(long_text)}")
    messages = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": USR.format(long_text=long_text)}
    ]
    response = process_request(messages)
    print(response)
    desc_path = "./description.txt"
    with open(desc_path, 'w', encoding='utf-8') as f:
        f.write(response)
        print(f"Description saved to {desc_path}")
