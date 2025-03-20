import json
from utils import *

raw_data_path = "./filtered_corpus.json"
queries_path = "./questions.json"
SYS = """You are a helpful assistant who can answer the user query according to the long text document. Provide detailed and accurate information based on the user's questions, ensuring that the responses are relevant and informative."""
USR = """Here is the long text document: \n\n{long_text} \n\nHere is the user query: \n{query} \nPlease provide the answer to the user query."""

with open(raw_data_path, 'r', encoding='utf-8') as f:
    long_text_raw = json.load(f)
    long_text = [f"Title: {l['title']} \nContent: {l['body']}" for l in long_text_raw]
    long_text = [remove_unicode_escapes(lt) for lt in long_text]
    long_text = [clean_non_utf8_chars(lt) for lt in long_text]
    long_text = [remove_surrogates(lt) for lt in long_text]
    long_text = "\n\n".join(long_text)
    

with open(queries_path, 'r', encoding='utf-8') as f:
    queries_raw = json.load(f)
    queries = [q["question"] for q in queries_raw]
    print(f"Number of queries: {len(queries)}")

all_messages = [[
    {"role": "system", "content": SYS},
    {"role": "user", "content": USR.format(long_text=long_text, query=query)}
] for query in queries]

all_responses = multi_process_request(all_messages, 1, process_request)
# Save the responses together with the queries to responses.json
for idx, q in enumerate(queries_raw):
    q["response"] = all_responses[idx]

save_to_json(queries_raw, "./long_context_responses_news.json")
