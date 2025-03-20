import json
from utils import *
from pydantic import BaseModel

class User(BaseModel):
    description: str

class AllUsers(BaseModel):
    users: list[User]

desc_path = "./description.txt"
SYS = """You are tasked with defining five distinct user personas who would undoubtedly be interested in the information contained in the given dataset. These personas must be clearly differentiated from one another, ensuring minimal overlap in their characteristics. The dataset description provides insights that will help you shape these personas. When crafting each persona, focus on the following:

1. **Demographic Information:** Age, gender, profession.
2. **Primary Interests:** What topics are they passionate about?
3. **Problem-Solving Goals:** What specific problems or challenges are they trying to solve with the information in this dataset?

Each persona should represent a user who would derive substantial value from the dataset, but the motivations and characteristics should vary significantly across the five personas. Prevent inluding specific entities or names in the persona descriptions to maintain a general and abstract representation."""
USR = """Here is the description of the dataset: \n{description} \nPlease define the users now."""

with open(desc_path, 'r', encoding='utf-8') as f:
    description = f.read()
    messages = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": USR.format(description=description)}
    ]
    response = process_request_structered(messages, AllUsers)
    print("\n\n".join([user.description for user in response.users]))
    users_desc = [user.description for user in response.users]

# Save the user descriptions to users.json
with open("./users.txt", 'w') as f:
    for user in users_desc:
        f.write(user + '\n')
    print(f"Users saved to users.txt")

class Task(BaseModel):
    description: str

class AllTasks(BaseModel):
    tasks: list[Task]

SYS = """You will be provided with a detailed user persona and the description of a dataset. Your task is to design five different Q&A-style tasks that are tailored specifically to this persona’s unique needs, interests, and goals. These tasks should help the user gain a deep understanding of the dataset from various angles.

For each Q&A task, focus on:

1. **Persona Alignment:** Ensure that the task is directly relevant to the persona’s interests and objectives, addressing their specific challenges or areas of curiosity.
2. **Variety of Perspectives:** Approach the dataset from different angles to ensure the user gains a comprehensive understanding of the information.
3. **Short and Clear:** Keep the tasks description within one sentence, making it easy for the user to understand and engage with. For example, "Understanding how tech leaders view the role of policy and regulation" or "Exploring the impact of climate change on global food security."

The five tasks should be distinct from one another, providing a well-rounded exploration of the dataset while aligning closely with the persona's profile."""
USR = """Here is the description of the dataset: \n{data_desc} \n\nHere is the user persona: \n{description} \nPlease define the tasks now."""

outputs = []
all_messages = [[
    {"role": "system", "content": SYS},
    {"role": "user", "content": USR.format(data_desc=description, description=users_desc_)}
] for users_desc_ in users_desc]
for idx, messages in enumerate(all_messages):
    response = process_request_structered(messages, AllTasks)
    print("\n\n".join([task.description for task in response.tasks]))
    tasks_desc = [task.description for task in response.tasks]
    outputs.append({
        "tasks": tasks_desc,
        "users": users_desc[idx]
    })

with open("./tasks.json", 'w') as f:
    json.dump(outputs, f, indent=4)
    print(f"Tasks saved to tasks.json")
