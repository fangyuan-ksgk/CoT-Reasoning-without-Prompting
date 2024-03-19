# LangGraph-Based Conversation Simulator

from typing import List
import openai
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI
from langchain.adapters.openai import convert_message_to_dict
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, MessageGraph
from langchain_core.messages.ai import AIMessage
from gen_prompt import *
import json
import re
import anthropic


def chat_bot_node(messages):
    # Convert from LangChain format to the OpenAI format, which our chatbot function expects.
    messages = [convert_message_to_dict(m) for m in messages]
    # Call the chat bot
    chat_bot_response = my_chat_bot(messages) # My Chat Bot things are essentially OpenAI, so we need the respective format of that message from the Langchain agent
    # Respond with an AI Message
    return AIMessage(content=chat_bot_response["content"])

def _swap_roles(messages): # Swap between AIMessage & HumanMessage
    new_messages = []
    for m in messages:
        if isinstance(m, AIMessage):
            new_messages.append(HumanMessage(content=m.content))
        else:
            new_messages.append(AIMessage(content=m.content))
    return new_messages


def simulated_user_node(messages):
    # Swap roles of messages
    new_messages = _swap_roles(messages)
    # Call the simulated user
    response = simulated_user.invoke({"messages": new_messages})
    # This response is an AI message - we need to flip this to be a human message
    return HumanMessage(content=response.content)


def should_continue(messages):
    if len(messages) > 6:
        return "end"
    elif messages[-1].content == "FINISHED":
        return "end"
    else:
        return "continue"
    

graph_builder = MessageGraph()
graph_builder.add_node("user", simulated_user_node)
graph_builder.add_node("chat_bot", chat_bot_node)
# Every response from  your chat bot will automatically go to the
# simulated user
graph_builder.add_edge("chat_bot", "user")
graph_builder.add_conditional_edges(
    "user",
    should_continue,
    # If the finish criteria are met, we will stop the simulation,
    # otherwise, the virtual user's message will be sent to your chat bot
    {
        "end": END,
        "continue": "chat_bot",
    },
)
# The input will first go to your chat bot, then the edge from chat_bot to user is set, whereas we then let the user to decide 
# whether it needs to return to the next round of chatting with the chat bot, next. 
graph_builder.set_entry_point("chat_bot")
simulation = graph_builder.compile()


def collect_conversation(chunk):
    conversation = []
    messages = chunk['__end__']
    for i in range(0, len(messages)):
        if isinstance((messages[i]), AIMessage):
            name = 'Sales'
            content = messages[i].content
            response = name+": "+content
        elif messages[i].content != 'FINISHED':
            name = "Customer"
            content = messages[i].content
            response = name+": "+content
        else:
            continue
        conversation.append(response)
    return conversation


def simulate_conversation(simulation=simulation):
    # Create a simulation with a single generator
    for chunk in simulation.stream([]):
        # Print out all events aside from the final end chunk
        1 + 1
        # if END not in chunk:
        #     print(chunk)
        #     print("----")
    conversation = collect_conversation(chunk)
    return conversation


def collect_ideas(message):
    """
    Collect Ideas from Anthropic on how to make the customer appears more angry
    """
    ideas = []
    # Assuming the response is stored in a variable called 'response'
    # response = "<document_content>...</document_content>"  # Replace with the actual response
    response = message.content[0].text

    # Extract the JSON data from the response
    match = re.search(r'```json(.*?)```', response, re.DOTALL)
    if match:
        json_data = match.group(1).strip()
        
        # Parse the JSON data
        data = json.loads(json_data)
        
        # Access the parsed data
        for option in data:
            # print(option)
            if isinstance(option, dict):
                dict_as_string = ", ".join([f"{key}: {value}" for key, value in option.items()])
                ideas.append(dict_as_string)
            else:
                ideas.append(str(option))
    else:
        print("No JSON data found in the response.")
    return ideas



def generate_revision_ideas(revision_idea_generation_prompt):
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=2000,
        temperature=0,
        messages=[{"role": "user", "content": revision_idea_generation_prompt}]
    )
    return collect_ideas(message)



