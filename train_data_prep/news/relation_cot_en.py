from utils import *
import json
import random
from pydantic import BaseModel

final_relations = load_parquet("../artifacts/create_final_relationships.parquet")
final_relations = final_relations[["source", "target", "description", "text_unit_ids"]]
final_relations = final_relations.to_dict(orient="records")
final_text_units = load_parquet("../artifacts/create_final_text_units.parquet")
final_text_units = final_text_units[["id", "text"]]
context_limit = 30

for relation in final_relations:
    text_unit_ids = relation["text_unit_ids"]
    relation["chunks"] = []
    for text_unit_id in text_unit_ids:
        text = final_text_units[final_text_units["id"] == text_unit_id]["text"].values[0]
        relation["chunks"].append(text)

templates = [
    "What do you know about {source_name} and {target_name}?",
    "Are you familiar with {source_name} and {target_name}?",
    "Do you know about {source_name} and {target_name}?",
    "What is your knowledge about {source_name} and {target_name}?",
    "What is the relationship between {source_name} and {target_name}?",
    "What is the connection between {source_name} and {target_name}?",
    "What is the association between {source_name} and {target_name}?",
    "Can you tell me about the relationship between {source_name} and {target_name}?",
    "Tell me about the relationship between {source_name} and {target_name}.",
]

relaQA_pairs_without_text_units = []

for relation in final_relations:
    source_name = relation["source"]
    target_name = relation["target"]
    description = relation["description"]
    template = random.choice(templates)
    relaQA_pairs_without_text_units.append({"user": template.format(source_name=source_name, target_name=target_name), "assistant": description})

with open("../output/relaQA_pairs_without_text_units.json", "w") as f:
    json.dump(relaQA_pairs_without_text_units, f, ensure_ascii=False, indent=4)

# form COT data
class COT_Relations(BaseModel):
    question: str
    cot_relation_answer: str

SYS = """**Generate a high-quality question-answer pair based on the provided relationship information. Below are the specific requirements:**

1. **The relationship information includes:**
    - **Entity 1** (source)
    - **Entity 2** (target)
    - **Relationship description** (relation_description)
    - **Context information related to the relationship** (context)

2. **Question Generation Rules:**
    - The generated question must include the names of Entity 1 and Entity 2 and should inquire about the relationship itself.
    - The content of the question must be based on the relationship description and related context information, with a high level of quality that encourages deep thinking or provides valuable insights.
    - Example: If the relationship is between A and B, the question can be "What is the connection between [A] and [B]?" or "Why is the relationship between [A] and [B] so important?"

3. **Answer Generation Rules:**
    - The answer should be constructed using a chain of thought (CoT) reasoning approach, starting with a restatement of the context information for thought and reasoning, and then incorporating the relationship description.
    - The context information should be rewritten in a high-quality manner to ensure that it flows naturally and conforms to language standards. The reasoning chain should include as much information as possible, and you can directly quote high-quality context information if needed.
    - The relationship description and context information must be closely integrated as intermediate information, supporting the final answer.
    - The final answer should be well-structured and based on the reasoning process described above, providing a comprehensive response to the question.
    - If the relationship description and context information are limited, the answer should be at least 300 words; if the relationship description and context information are sufficient, the answer should be at least 500 words.
    - Use Markdown format for the response. For example, use subheadings, bullet points, etc., to make the answer well-organized and easy to read. """
USR = """Here is the relationship information:

Entity 1: {source}
Entity 2: {target}
Relationship Description: {relation_description}
Context Information: 
{context}

Please generate a question-answer pair based on the information provided above.
"""

all_cot_relation_messages = []
for relation in final_relations:
    source = relation["source"]
    target = relation["target"]
    relation_description = relation["description"]
    context = format_context(relation["chunks"][:context_limit])
    all_cot_relation_messages.append([{
        "role": "system",
        "content": SYS
    }, {
        "role": "user",
        "content": USR.format(source=source, target=target, relation_description=relation_description, context=context)
    }])

cot_relation_res = multi_process_request(all_cot_relation_messages, 5, process_request_structered, COT_Relations)
all_qa_pairs = [{"user": cot_relation.question, "assistant": cot_relation.cot_relation_answer} for cot_relation in cot_relation_res if type(cot_relation) != str]

with open("../output/relaQA_pairs_with_text_units.json", "w") as f:
    json.dump(all_qa_pairs, f, ensure_ascii=False, indent=4)
