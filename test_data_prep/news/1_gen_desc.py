import json
from utils import *

long_text_dir = "./filtered_corpus.json"
SYS = """You are an professional agent to summarize a long text document. Organize the summary logically, with clear sections corresponding to the different categories. Focus on the summarization of each category rather than specific details or articles.
"""
USR = """Here is a long text document that needs to be summarized:

{long_text}

Please generate a comprehensive summary of the document according to the provided guidelines.
"""
with open(long_text_dir, 'r', encoding='utf-8') as f:
    long_text = json.load(f)
    long_text = [f"Category: {l['category']} \nTitle: {l['title']} \nAuthor: {l['author']} \nSource: {l['source']} \nContent: {l['body']}" for l in long_text]
    long_text = [remove_unicode_escapes(lt) for lt in long_text]
    long_text = [clean_non_utf8_chars(lt) for lt in long_text]
    long_text = [remove_surrogates(lt) for lt in long_text]
    print("Text length", len('\n\n'.join(long_text)))
    messages = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": USR.format(long_text="\n\n".join(long_text))}
    ]
    response = process_request(messages)
    print(response)
    desc_path = "./description.txt"
    with open(desc_path, 'w', encoding='utf-8') as f:
        f.write(response)
        print(f"Description saved to {desc_path}")
