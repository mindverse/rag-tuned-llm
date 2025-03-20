from utils import *
import json
import random
from pydantic import BaseModel

class Thought_and_Question(BaseModel):
    thought: str
    question: str

final_text_units = load_parquet("../artifacts/create_final_text_units.parquet")
final_entities = load_parquet("../artifacts/create_final_entities.parquet")
final_relationships = load_parquet("../artifacts/create_final_relationships.parquet")

# subset final_entities using id, name, type, description, and text_unit_ids
final_entities = final_entities[["id", "name", "description"]]
# subset final_text_units using id, text, entity_ids and relationship_ids
final_text_units = final_text_units[["id", "text", "entity_ids", "relationship_ids"]]
# subset final_relationships using id, source, target, description
final_relationships = final_relationships[["id", "source", "target", "description"]]

# transfer the final_text_units to a list of dictionaries, and use each dictionary's entity_ids and relationship_ids to get the corresponding entities from final_text_units and relationships from final_relationships
final_text_units = final_text_units.to_dict(orient="records")
for text_unit in final_text_units:
    entity_ids = text_unit["entity_ids"]
    relationship_ids = text_unit["relationship_ids"]
    text_unit["entities"] = []
    text_unit["relationships"] = []
    if entity_ids is not None: 
        for entity_id in entity_ids:
            entity = final_entities[final_entities["id"] == entity_id]
            text_unit["entities"].append(entity)
    if relationship_ids is not None:
        for relationship_id in relationship_ids:
            relationship = final_relationships[final_relationships["id"] == relationship_id]
            text_unit["relationships"].append(relationship)

# form COT data
SYS_Q = """# Role #
You are a question-generation assistant. Your task is to generate a question based on a text chunk and some related information (entity in the text chunk or relationship between the entities in the text chunk). The question should focus on the entities or relationship mentioned in the text chunk and reflect a broader perspective. Ensure that the question you generate can be answered, at least in part, based on the information provided.

# Workflow #
**Step 1**: You will be provided with a text chunk and some related information, including entities and relationships. First, assess whether there is enough valuable information inside the given information to allow you to generate a question with a broader, global scope. 
**Step 2**: If the information is insufficient to generate meaningful question, return "Unable to generate." If it is sufficient, proceed by returning your question in the format described below.

# Guidelines #
1. Ensure that the question can be answered based on the information provided (text chunk, entity description, relationship description). The question should be meaningful and reasonable.
2. Make sure the generated question relevant to the entities or relationship mentioned.
3. The question should have a broader, global nature, meaning they require combining insights from multiple information sources to answer.
4. If Step 1 confirms sufficient connection, generate a high-quality question.

# Response Format #

thought: Your thought process  
question: xxx / Unable to generate
"""
USR_Q_ENTITY = """Here is a text chunk:
{text}

Here is the information about an entity mentioned in the text:
{entity}

Based on this information, please generate a high-quality question related to the entity following the guidelines provided.
"""
USR_Q_RELATIONSHIP = """Here is a text chunk:
{text}

Here is the information about a relationship mentioned in the text:
{relationship}

Based on this information, please generate a question related to the relationship following the guidelines provided.
"""

all_questions_messages = []
for text_unit in final_text_units:
    text = text_unit["text"]
    entities = text_unit["entities"]
    for entity in entities:
        entity_name = entity["name"].values[0]
        entity_description = entity["description"].values[0]
        message_usr = USR_Q_ENTITY.format(text=text, entity=f"Entity Name: {entity_name}\nEntity Description: {entity_description}")
        all_questions_messages.append({"message": [{
            "role": "system",
            "content": SYS_Q
        }, {
            "role": "user",
            "content": message_usr
        }], "text_unit_id": text_unit["id"], "entity_id": entity["id"].values[0], "relationship_id": None})
    relationships = text_unit["relationships"]
    for relationship in relationships:
        relationship_description = relationship["description"].values[0]
        message_usr = USR_Q_RELATIONSHIP.format(text=text, relationship=f"Relationship: between {relationship['source'].values[0]} and {relationship['target'].values[0]}\nRelationship Description: {relationship_description}")
        all_questions_messages.append({"message": [{
            "role": "system",
            "content": SYS_Q
        }, {
            "role": "user",
            "content": message_usr
        }], "text_unit_id": text_unit["id"], "relationship_id": relationship["id"].values[0], "entity_id": None})
print("Number of messages", len(all_questions_messages))

results = multi_process_request([r["message"] for r in all_questions_messages], 2, process_request_structered, Thought_and_Question)
all_pairs = [{"thought": result.thought, "question": result.question, "text_unit_id": r['text_unit_id'], "entity_id": r['entity_id'], "relationship_id": r['relationship_id']} for result, r in zip(results, all_questions_messages) if type(result) != str]
# save the results to a json file
save_to_json(all_pairs, "../output/textunits.json")

