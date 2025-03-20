import openai
from concurrent.futures import ThreadPoolExecutor
import json
from tqdm import tqdm
from utils import *

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_API_BASE_URL")
MODEL = os.getenv("OPENAI_API_MODEL")

def filter(messages):
    client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
    model = MODEL
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content

long_text_dir = "./corpus.json"
SYS = """You are a data checker. Whatever you read, just response with '1'."""
with open(long_text_dir, 'r', encoding='utf-8') as f:
    long_text_raw = json.load(f)
    long_text = [f"Title: {l['title']} \nContent: {l['body']}" for l in long_text_raw]
    long_text = [remove_unicode_escapes(lt) for lt in long_text]
    long_text = [clean_non_utf8_chars(lt) for lt in long_text]
    long_text = [remove_surrogates(lt) for lt in long_text]
    messages = [[{"role": "system", "content": SYS}, {"role": "user", "content": tmp_text}] for tmp_text in long_text]

    with ThreadPoolExecutor(max_workers=min(1, len(messages))) as executor:
            futures = [executor.submit(filter, message) for message in messages]
            results = []

            for idx, future in enumerate(tqdm(futures)):
                try:
                    result = future.result()
                    results.append(long_text_raw[idx])

                except Exception as e:
                    print(f"Raise ERROR: {e} WHEN GENERATE RESPONSE")

    filtered_path = "./filtered_corpus.json"
    with open(filtered_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Filtered corpus saved to {filtered_path}")

