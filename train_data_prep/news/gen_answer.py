from utils import *
import json
import random
from pydantic import BaseModel

class Thought_and_Answer(BaseModel):
    thought: str
    answer: str

final_text_units = load_parquet("../artifacts/create_final_text_units.parquet")
final_entities = load_parquet("../artifacts/create_final_entities.parquet")
final_relationships = load_parquet("../artifacts/create_final_relationships.parquet")

# subset final_entities using id, name, type, description, and text_unit_ids
final_entities = final_entities[["id", "name", "description"]]
# subset final_text_units using id, text, entity_ids and relationship_ids
final_text_units = final_text_units[["id", "text", "entity_ids", "relationship_ids"]]
# subset final_relationships using id, source, target, description
final_relationships = final_relationships[["id", "source", "target", "description"]]

# load the questions which have been generated
questions = "../output/textunits.json"
with open(questions) as f:
    questions = json.load(f)

# form COT data
SYS_A = """# Role #
You are a text-answering assistant. Your task is to generate an answer based on a question, thoughts of generating this question, a text chunk about the question, and the related entity/relation information. The answer should be clear, concise, and cover all the necessary aspects of the question using the provided information. Make sure the answer is relevant to the entities or relationships mentioned in the question.

# Workflow #
**Step 1**: You will be provided with a question, the thought process behind generating the question, a text chunk, and related information, such as entities or relationships. 
First, think deeply about how to answer the question using the provided information. Make sure that the thought reflects a global or broader perspective if necessary.
**Step 2**: If there is insufficient information to answer the question fully, provide a partial answer and acknowledge that additional details may be required.
**Step 3**: Structure your response in a clear, logical format that synthesizes the information effectively. You may combine insights from multiple sources (text chunk, entity descriptions, relationships) to form a well-rounded response.

# Guidelines #
1. Ensure the answer fully or partially addresses the question, based on the available information (text chunk, entity description, relationship description).
2. Make sure the answer aligns with the entities or relationships mentioned.
3. The answer should be meaningful and reasonable, balancing a broader, global perspective with the specifics provided.
4. If necessary, acknowledge any gaps in the information, noting where additional insights might be required.

# Response Format #

thought: Your thought process  
answer: xxx"""
USR_A_ENTITY = """Here is a text chunk:
{text}

Here is the information about an entity mentioned in the text:
{entity}

Here is a question generated based on the text chunk and the entity information:
{question}

Here is the thought process behind generating the question:
{thought}

Based on the text chunk and the entity information, please provide the answer as well as your thought process."""
USR_A_RELATIONSHIP = """Here is a text chunk:
{text}

Here is the information about a relationship mentioned in the text:
{relationship}

Here is a question generated based on the text chunk and the relationship information:
{question}

Here is the thought process behind generating the question:
{thought}

Based on the text chunk and the relationship information, please provide the answer as well as your thought process."""

all_answers_messages = []
for question in questions:
    thought = question["thought"]
    q = question["question"]
    text_unit_id = question["text_unit_id"]
    entity_id = question["entity_id"]
    relationship_id = question["relationship_id"]
    text_unit = final_text_units[final_text_units["id"] == text_unit_id].iloc[0]
    text = text_unit["text"]
    if entity_id is not None:
        entity = final_entities[final_entities["id"] == entity_id].iloc[0]
        entity_text = f"Entity Name: {entity['name']}\nEntity Description: {entity['description']}"
        usr_text = USR_A_ENTITY.format(text=text, entity=entity_text, question=q, thought=thought)
        all_answers_messages.append({"message": [{
            "role": "system",
            "content": SYS_A
        }, {
            "role": "user",
            "content": usr_text
        }], "text_unit_id": text_unit_id, "entity_id": entity_id, "relationship_id": None, "question": q})
    elif relationship_id is not None:
        relationship = final_relationships[final_relationships["id"] == relationship_id].iloc[0]
        relationship_text = f"Source: {relationship['source']}\nTarget: {relationship['target']}\nDescription: {relationship['description']}"
        usr_text = USR_A_RELATIONSHIP.format(text=text, relationship=relationship_text, question=q, thought=thought)
        all_answers_messages.append({"message": [{
            "role": "system",
            "content": SYS_A
        }, {
            "role": "user",
            "content": usr_text
        }], "text_unit_id": text_unit_id, "relationship_id": relationship_id, "entity_id": None, "question": q})
    else:
        continue

print("Number of messages", len(all_answers_messages))

results = multi_process_request([r["message"] for r in all_answers_messages], 15, process_request_structered, Thought_and_Answer)
all_pairs = [{"assistant": result.thought + result.answer, "text_unit_id": r['text_unit_id'], "entity_id": r['entity_id'], "relationship_id": r['relationship_id'], "user": r['question']} for result, r in zip(results, all_answers_messages) if type(result) != str]

# save the results to a json file
save_to_json(all_pairs, "../output/textunits_answers.json")
