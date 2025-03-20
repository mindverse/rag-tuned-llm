import json
import openai
import random
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd
import os

def load_parquet(file_path):
    """
    Load a parquet file.
    """
    return pd.read_parquet(file_path)

def process_request(messages):
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE_URL"))
    model = "gpt-4o-mini"
    try: 
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        out = parse_res(response.choices[0].message.content)
        return out
    except:
        # retry for 3 times
        for i in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                out = parse_res(response.choices[0].message.content)
                return out
            except:
                pass
        print(f"Raise ERROR WHEN GENERATE RESPONSE")
        return "Failed to generate the response."

def map_ranking(ranking):
    if "A" in ranking:
        return [1, 2]
    elif "B" in ranking:
        return [2, 1]
    else:
        return [1, 1]

def parse_res(evaluation):
    eval_results = evaluation.replace("*", "").split("\n\n")
    eval_results = [er for er in eval_results if er.replace("\n", "").strip() != ""]
    eval_results = [eval_result.split("\n") for eval_result in eval_results]
    eval_results = [[x for x in er if x.replace(" ", "").replace("\n", "") != ""] for er in eval_results]
    eval_result_list = []
    for eval_result in eval_results:
        aspect = eval_result[0].split(": ")[0]
        ranking = eval_result[0].split(": ")[1]
        reason = eval_result[1].split(": ")[1]
        eval_result_list.append({
            "aspect": aspect,
            "ranking": map_ranking(ranking),
            "reason": reason
        })
    return eval_result_list

# load all the competitors and the golden label
infer_dir = "" # specify the path to the model results
news_C0 = json.load(open("news_local_search_c0_full.json"))
# news_C1 = json.load(open("news_local_search_c1_full.json"))
# news_C2 = json.load(open("news_local_search_c2_full.json"))
# news_C3 = json.load(open("news_local_search_c3_full.json"))
# news_naiverag = json.load(open("news_naiverag_results.json"))
news_model = json.load(open(infer_dir))
# load context information
news_instances = json.load(open("news_local_search_c1_full.json"))

# prompts
SYS = """You are tasked with comparing two answers provided in response to a given question and the context information. Follow these guidelines:

1. Careful Reading: Begin by carefully reading the original question, the context information, and the two responses from our competitors.
2. Scoring Criteria: Evaluate which answer does better in terms of the following dimensions:
    - Usefulness / Correctness: How accurate and reliable is the answer? Does it directly address the question and provide useful information?
    - Richness / Diversity: How varied and detailed is the content? Does the answer cover different perspectives or provide in-depth explanations?
    - Insightfulness / Deep Understanding: Does the answer demonstrate a deep comprehension of the topic? Does it offer thoughtful or original insights?
    - User-Friendliness: How easy is the answer to read and understand? Is it well-structured, concise, and accessible to the reader?

3. Explanation: After evaluating the answers based on the criteria, provide a detailed explanation for your rankings. Justify your assessment with specific reasons.

Output Format: (Assume the answers are labeled A and B)

Usefulness / Correctness: A/B
Reason: ...
Richness / Diversity: A/B
Reason: ...
Insightfulness / Deep Understanding: A/B
Reason: ...
User-Friendliness: A/B
Reason: ...

Ensure that your evaluation and explanations are well-supported by your analysis of the content. Your evaluation should be thorough, fair, and based on the quality of the answers provided. To stress, a good answer is not always longer, but more logical."""
USR = """Here is the question:
{question}

Here are two answers from our competitors:
-- start of answer A --
{answers}
-- end of answer B --

Here is the context information:
-- start of context --
{context}
-- end of context --

Respond according to the guidelines provided, do not include any other information in the response. Stricly follow the format provided above."""

news_messages = []
all_news_answers = []
# orders_news = generate_balanced_list(len(news_model), 2)
prefix = ['A: ', 'B: ']

for news_c0, news_m, indx in zip(news_C0, news_model, range(len(news_model))):
    question = news_m.get("user", news_m.get("question", news_m.get("response", '')))
    answers = [news_m.get("assistant", news_m.get("response", '')), news_c0.get("response", news_c0.get("answer", ''))]
    # shuffle the answers and remember the order
    # order = orders_news[indx]
    # answers = [answers[i] for i in order]
    answers = [x + y for x, y in zip(prefix, answers)]
    
    context = news_instances[indx]['context_text'][:50000]
    news_messages.append([{
        "role": "system",
        "content": SYS
    }, {
        "role": "user",
        "content": USR.format(question=question, answers="\n-- end of answer A --\n\n-- start of answer B --\n".join(answers), context=context)
    }])
    all_news_answers.append("\n\n".join(answers))
    # orders_news.append(order)

all_evals_news = []
with ThreadPoolExecutor(max_workers=min(4, len(news_messages))) as executor:
    futures = [(i, executor.submit(process_request, messages)) for i, messages in enumerate(news_messages)]
    eval_results = [None] * len(news_messages)

    for i, future in tqdm(futures):
        try:
            evaluation = future.result()
            if evaluation != "Failed to generate the response.":
                eval_results[i] = {"question": news_model[i].get("user", news_model[i].get("question", '')),
                                   "eval_result": evaluation, 
                                   "all_answers": all_news_answers[i], 
                                   "order": [0, 1]}
            else:
                eval_results[i] = None
        except Exception as e:
            eval_results[i] = f"Raise ERROR: {e} WHEN GENERATE RESPONSE"
all_evals_news = [result for result in eval_results if result is not None]
print(f"Successfully generated {len(all_evals_news)} responses.")

# save the results
with open(infer_dir.replace('.json', '_res.json'), "w") as f:
    json.dump(all_evals_news, f, indent=4)
