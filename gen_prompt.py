from typing import List
import openai
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI

copilot_prompt = """You are a FWD life insurance agent. You will be talking to a customer.

{instructions}
"""
copilot_instruction = """Do not act as the customer, stick to your role, only respond as the insurance agent. Keep your response short and conversational."""
copilot_prompt = copilot_prompt.format(instructions=copilot_instruction)

pilot_prompt_template = """You are a customer, having a conversation with a FWD life insurance agent. \

{instructions}

When you are finished with the conversation, respond with a single word 'FINISHED'"""

pilot_instruction = """Your name is Harrison. Keep your response short and conversational."""

script_writter_prompt = """Based on the revision idea, re-write the conversation following the same format, as a list of string. \

Revision Idea: 
{revision_idea}
"""


def script_rewriter(revision_idea, conversation):
    system_prompt = script_writter_prompt.format(revision_idea=revision_idea)
    prompt = """Here is the conversation to be rewritten with the revision idea. Please keep the same format in your output. \
        
        Conversation:
        {conversation}
        """
    system_message = {
        "role": "system",
        "content": script_writter_prompt.format(revision_idea=revision_idea)
    }
    user_message = {
        "role": "user",
        "content": prompt
    }
    messages = [system_message, user_message]
    completion = openai.chat.completions.create(
        messages=messages, model="gpt-4"
    )
    return completion.choices[0].message.model_dump()



# This is flexible, but you can define your agent here, or call your agent API here.
def my_chat_bot(messages: List[dict]) -> dict:
    system_message = {
        "role": "system",
        "content": copilot_prompt,
    }
    messages = [system_message] + messages
    completion = openai.chat.completions.create(
        messages=messages, model="gpt-3.5-turbo"
    )
    return completion.choices[0].message.model_dump()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", pilot_prompt_template),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

prompt = prompt.partial(instructions=pilot_instruction)

model = ChatOpenAI()

simulated_user = prompt | model


revision_idea_prompt = """
Here is a conversation between sales and customers, provide json format output on {num_of_ideas} possible different ways to {revision_goal}. Give your suggestion very concisely. Remember this is a conversation, not a phone call.

{conversation}

Give your output in json format
"""