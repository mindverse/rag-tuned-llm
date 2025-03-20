from utils import *
import json
import random
from pydantic import BaseModel

final_entities = load_parquet("../artifacts/create_final_entities.parquet")
final_text_units = load_parquet("../artifacts/create_final_text_units.parquet")
context_limit = 30

# subset final_entities using name, type, description, and text_unit_ids
final_entities = final_entities[["name", "type", "description", "text_unit_ids"]]
# subset text units using id and text
final_text_units = final_text_units[["id", "text"]]

# print the distribution of entity types
print(final_entities["type"].value_counts())
# only keep entities with type "EVENT", "ORGANIZATION", "PERSON", "GEO", "FOOD"
final_entities = final_entities[final_entities["type"].isin(["EVENT", "ORGANIZATION", "PERSON", "GEO", "FOOD"])]
# transfer the final_entities to a list of dictionaries, and use each dictionary's text_unit_ids to get the corresponding text from final_text_units, use "chunks" to store the texts
final_entities = final_entities.to_dict(orient="records")
for entity in final_entities:
    text_unit_ids = entity["text_unit_ids"]
    entity["chunks"] = []
    for text_unit_id in text_unit_ids:
        text = final_text_units[final_text_units["id"] == text_unit_id]["text"].values[0]
        entity["chunks"].append(text)

# templates for different entity types
templates = {
    "EVENT": [
        "What can you tell me about the event {entity_name}?",
        "Have you heard about the event {entity_name}? If so, what do you know?",
        "What insights do you have regarding the event {entity_name}?",
        "Could you share some information about {entity_name}?",
        "I’d like to learn more about {entity_name}. Can you help?",
        "What details do you know about {entity_name}?"
    ],
    "ORGANIZATION": [
        "What can you share about the organization {entity_name}?",
        "Are you familiar with {entity_name}? What do you know?",
        "Can you tell me more about the organization called {entity_name}?",
        "What have you heard or read about {entity_name}?",
        "Do you have any insights on {entity_name} as an organization?",
        "Could you explain what {entity_name} does?"
    ],
    "PERSON": [
        "What do you know about {entity_name} as a person?",
        "Can you tell me about {entity_name}?",
        "How familiar are you with {entity_name}? What can you share?",
        "Have you come across {entity_name} before? What do you know?",
        "I’d love to learn more about {entity_name}. What can you tell me?",
        "What insights do you have about {entity_name}?"
    ],
    "GEO": [
        "What do you know about the place {entity_name}?",
        "Can you share information about {entity_name} as a location?",
        "What’s your knowledge of {entity_name}?",
        "How much do you know about {entity_name}? What can you share?",
        "I’m curious about {entity_name}. What can you tell me?",
        "What’s notable about {entity_name}?"
    ],
    "FOOD": [
        "What do you know about the dish {entity_name}?",
        "Have you tried {entity_name} before? Can you tell me more?",
        "What can you share about {entity_name} as a food?",
        "Could you explain what {entity_name} is like as a dish?",
        "What’s special about the food {entity_name}?",
        "What’s your knowledge of {entity_name}?"
    ]
}

QA_pairs_without_text_units = []
# generate QA pairs for each entity, chose a template randomly, and replace {entity_name} with the entity's name, and use description as the answer
for entity in final_entities:
    entity_name = entity["name"]
    entity_type = entity["type"]
    description = entity["description"]
    template = random.choice(templates[entity_type])
    QA_pairs_without_text_units.append({"user": template.format(entity_name=entity_name), "assistant": description})

# save the QA pairs to a json file
with open("../output/QA_pairs_without_text_units.json", "w") as f:
    json.dump(QA_pairs_without_text_units, f, ensure_ascii=False, indent=4)

# form COT data
class COT_Data(BaseModel):
    question: str
    cot_answer: str

SYS = """**Generate a high-quality question-answer pair based on the provided entity information. Below are the specific requirements:**

1. **The entity information includes:**
   - **Entity Name** (entity_name)
   - **Entity Type** (entity_type)
   - **Entity Description** (entity_description)
   - **Context information related to the entity** (context)

2. **Question Generation Rules:**
   - The generated question must include the entity name and should inquire about the entity itself.
   - The content of the question should be based on the entity description and relevant context information, with a high level of quality that encourages deep thinking or provides valuable insights.
   - Example: If the entity is a person, the question could be, "What kind of person is [entity_name]?" or "Why is [entity_name] so important in [context]?"

3. **Answer Generation Rules:**
   - The answer should be constructed using a chain of thought (CoT) reasoning approach, starting with a restatement of the context information for thought and reasoning, followed by incorporating the entity description.
   - The context information should be rewritten in a high-quality manner to ensure natural language flow and alignment with language standards. The reasoning chain should include as much information as possible, and you can directly quote high-quality context information if needed.
   - The entity description and context information must be closely integrated as intermediate information, supporting the final answer.
   - The final answer should be well-structured, based on the reasoning process described above, and provide a comprehensive response to the question.
   - If the entity description and context information are limited, the answer should be at least 300 words; if the entity description and context information are sufficient, the answer should be at least 500 words.
   - Use Markdown format for the response. For example, use subheadings, bullet points, etc., to make the answer well-organized and easy to read."""
USR = """Here is an entity's information:

Entity Name: {entity_name}
Entity Type: {entity_type}
Entity Description: {entity_description}
Context Information:
{entity_context}

Please generate a question-answer pair based on the information provided above.
"""

# generate the messages for the COT task
all_cot_messages = []
for entity in final_entities:
    entity_name = entity["name"]
    entity_type = entity["type"]
    entity_description = entity["description"]
    entity_context = format_context(entity["chunks"][:context_limit])
    all_cot_messages.append([{
        "role": "system",
        "content": SYS
    }, {
        "role": "user",
        "content": USR.format(entity_name=entity_name, entity_type=entity_type, entity_description=entity_description, entity_context=entity_context)
    }])
# multi-process the COT task
cot_results = multi_process_request(all_cot_messages, 15, process_request_structered, COT_Data)
# if fail, `Error occurred:` would be the response

# save the COT results to a json file
all_qa_pairs = [{"user": cot_result.question, "assistant": cot_result.cot_answer} for cot_result in cot_results if type(cot_result) != str]

with open("../output/QA_pairs_with_text_units.json", "w") as f:
    json.dump(all_qa_pairs, f, ensure_ascii=False, indent=4)
