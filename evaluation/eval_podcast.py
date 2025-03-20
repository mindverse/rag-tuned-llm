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
podcast_C0 = json.load(open("podcast_local_search_c0_full.json"))
# podcast_C1 = json.load(open("podcast_local_search_c1_full.json"))
# podcast_C2 = json.load(open("podcast_local_search_c2_full.json"))
# podcast_C3 = json.load(open("podcast_local_search_c3_full.json"))
# podcast_naiverag = json.load(open("podcast_naiverag_results.json"))
podcast_model = json.load(open(infer_dir))
# load context information
podcast_instances = json.load(open("podcast_local_search_c1_full.json"))

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

Usefulness / Correctness: A
Reason: ...
Richness / Diversity: A
Reason: ...
Insightfulness / Deep Understanding: A
Reason: ...
User-Friendliness: A
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

podcast_messages = []
all_podcast_answers = []
# orders_podcast = generate_balanced_list(len(podcast_model), 2)
prefix = ['A: ', 'B: ']

for podcast_c0, podcast_m, indx in zip(podcast_C0, podcast_model, range(len(podcast_model))):
    question = podcast_m.get("user", podcast_m.get("question", podcast_m.get("response", '')))
    answers = [podcast_m.get("assistant", podcast_m.get("response", '')), podcast_c0.get("response", podcast_c0.get("answer", ''))]
    # shuffle the answers and remember the order
    # order = orders_podcast[indx]
    # answers = [answers[i] for i in order]
    answers = [x + y for x, y in zip(prefix, answers)]
    
    context = podcast_instances[indx]['context_text'][:50000]
    podcast_messages.append([{
        "role": "system",
        "content": SYS
    }, {
        "role": "user",
        "content": USR.format(question=question, answers="\n-- end of answer A --\n\n-- start of answer B --\n".join(answers), context=context)
    }])
    all_podcast_answers.append("\n\n".join(answers))
    # orders_podcast.append(order)

all_evals_podcast = []
with ThreadPoolExecutor(max_workers=min(4, len(podcast_messages))) as executor:
    futures = [(i, executor.submit(process_request, messages)) for i, messages in enumerate(podcast_messages)]
    eval_results = [None] * len(podcast_messages)

    for i, future in tqdm(futures):
        try:
            evaluation = future.result()
            if evaluation != "Failed to generate the response.":
                eval_results[i] = {"question": podcast_model[i].get("user", podcast_model[i].get("question", '')),
                                   "eval_result": evaluation, 
                                   "all_answers": all_podcast_answers[i], 
                                   "order": [0, 1]}
            else:
                eval_results[i] = None
        except Exception as e:
            eval_results[i] = f"Raise ERROR: {e} WHEN GENERATE RESPONSE"
all_evals_podcast = [result for result in eval_results if result is not None]
print(f"Successfully generated {len(all_evals_podcast)} responses.")

# save the results
with open(infer_dir.replace('.json', '_res.json'), "w") as f:
    json.dump(all_evals_podcast, f, indent=4)
