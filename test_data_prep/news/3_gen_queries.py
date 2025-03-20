from utils import *
import json
from pydantic import BaseModel

class Questions(BaseModel):
    questions: list[str]

desc_path = "./description.txt"
SYS = """You will be provided with a dataset description. Based on the description, your task is to generate five questions that align with the following criteria:

1. The questions should be complex enough that a Retrieval Augmented Generation (RAG) model would struggle to answer them, but a well-trained Large Language Model (LLM) could.
  
2. Avoid mentioning specific entity names in the questions. Instead, the questions should require deep reasoning that combines real-world knowledge with insights derived from the dataset, not just simple fact retrieval.

3. Focus on macro-level analysis and trends rather than specific details. The questions should encourage thoughtful exploration of the dataset as a whole and require synthesis across different parts of the data. They should not have fixed, concrete answers.

4. The questions should guide the LLM to consider broader implications, hidden relationships, and nuanced patterns in the dataset that go beyond surface-level information.

Your task is to generate **five questions** based on these guidelines."""
USR = """Here is the description of the dataset: \n{data_desc}\n\nPlease generate the questions now."""

all_messages = []
with open(desc_path, 'r', encoding='utf-8') as f:
    description = f.read()
    for _ in range(50):
        all_messages.append([
            {"role": "system", "content": SYS},
            {"role": "user", "content": USR.format(data_desc=description)}
        ])

# Generate the questions
all_responses = multi_process_request(all_messages, 10, process_request_structered, Questions)
# print(all_responses)
all_questions = [query for response in all_responses for query in response.questions]
save_to_json(all_questions, "./questions_global.json")
