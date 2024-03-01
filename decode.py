# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import time
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from tqdm.notebook import tqdm
import re
from typing import Tuple, List

# model_name = "mistralai/Mistral-7B-v0.1"
# model_name = "mlabonne/Monarch-7B"
model_name = "HuggingFaceH4/zephyr-7b-beta"
# model_name = "mistralai/Mixtral-8x7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

""""
--- Cool Function ---
* Check the log probability of any response given a query | Given a huggingface Model & Tokenizer 
"""
def check_response_prob(model, tokenizer, query, target_response, max_required_tokens=80):
    inputs = tokenizer([query], return_tensors="pt")
    gen_out = model.generate(**inputs, output_scores=True, return_dict_in_generate=True, max_new_tokens=max_required_tokens, pad_token_id=tokenizer.eos_token_id)

    target_ids = tokenizer.encode(target_response)
    sum_of_logits = 0
    sum_of_probs = 0
    for i, id in enumerate(target_ids):
        prob = torch.nn.functional.softmax(gen_out.scores[i], dim=1)[0, id]
        sum_of_probs += prob
    avg_prob = sum_of_probs / len(target_ids)
    return avg_prob

check_token_length = lambda word: len(tokenizer(word).input_ids)
get_max_tokens_required = lambda words: max([check_token_length(word) for word in words])

def single_choice_response(model, tokenizer, query, possible_answers):
    max_tokens_required = get_max_tokens_required(possible_answers)
    unorm_probs = []
    for target_response in possible_answers:
        prob = check_response_prob(model, tokenizer, query, target_response=target_response, max_required_tokens=max_tokens_required)
        unorm_probs.append(prob)
    norm_const = sum(unorm_probs)
    probs = [prob / norm_const for prob in unorm_probs]
    return probs

query_template = """Compare customers' response in the two conversations: 
    Conversation A: {conversation_a}
    Conversation B: {conversation_b}
    {compare_query}
    Your answer: """

def pairmatch_open(conversation_pairs: Tuple[str, str],
                   compare_attributes: List[str],
                   query_template = query_template,
                   possible_answers = ["Yes", "No", "Unsure"],
                   model = model,
                   tokenizer = tokenizer) -> List[dict]:
    """
    Open-sourced LLM based Multi-Attribute Comparative Rater
    """
    results = []
    for compare_attribute in compare_attributes:
        query = query_template.format(compare_query=compare_attribute, conversation_a=conversation_pairs[0], conversation_b=conversation_pairs[1])
        result = single_choice_response(model, tokenizer, query, possible_answers)
        results.append(result)
        
    return results


"""
Now, I wish to do tree-search to locate the confident answer, and NOT the confident continuation
1. Should be doable by checking on the next-token
"""
def get_next_token_logit(model, tokenizer, query):
    inputs = tokenizer([query], return_tensors="pt")
    gen_out = model.generate(**inputs, output_scores=True, return_dict_in_generate=True, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
    return gen_out.scores[-1]

# Get Top-k next logits, then greedy-1 search afterwards
def get_k_branch(model, tokenizer, query, k=5):
    logit = get_next_token_logit(model, tokenizer, query)
    k_token = logit[0].argsort()[-k:]
    k_response = []
    for token in k_token:
        new_query = query + tokenizer.decode(token)
        candidate_inputs = tokenizer(new_query, return_tensor="pt")
        gen_out = model.generate(**candidate_inputs, output_scores=True, return_dict_in_generate=True)
        k_response.append(tokenizer.decode(gen_out.sequences[0], skip_special_tokens=True))
    return k_response

# Token Path Probability
def get_token_path_prob(gen_out, num_append:int = 1):
    logits = gen_out.scores
    num_output = len(logits)
    output_ids = gen_out.sequences[0][-num_output-num_append:]
    # output = tokenizer.decode(output_ids, skip_special_tokens=True)
    path_prob = torch.stack([score[0].max() for score in logits])
    path_prob = torch.nn.functional.softmax(path_prob, dim=0)
    # path_logprob = torch.log(path_prob)
    return output_ids, path_prob
    
# Word Path Probability -- Ensemble(word[token1,token2,...]) is the average probability of token appearance likelihood
def get_path_prob(gen_out, init_token_prob=None):
    if init_token_prob is None:
        token_ids, probs = get_token_path_prob(gen_out, num_append=0)
    else:
        token_ids, probs = get_token_path_prob(gen_out)
        probs = torch.concat([init_token_prob, probs])
    current_n_words = 0
    current_prob = 0
    word_probs = []
    ids = []
    current_n_tokens = 0
    word_prob = 0
    current_n_words = 0
    for token_id, prob in zip(token_ids, probs):
        ids.append(token_id)
        decode_seq = tokenizer.decode(ids)
        # print('Decode Sequence: ', decode_seq)
        words = re.split(r' |\n|\.\|:', decode_seq)
        # print('Splitted Words: ')
        # print(words)
        word = words[-1]
        if len(words) == current_n_words:
            word_prob += prob
            current_n_tokens += 1
            # more than one tokens correspond to the same word, word gets updated
            word_probs[-1] = (word, word_prob / current_n_tokens) # replace the previous word in the word prob list
        elif len(words) > current_n_words: # A old word is determined
            word_prob = prob
            current_n_tokens = 1
            word_probs.append((word, word_prob / current_n_tokens))
            current_n_words += 1
    return word_probs

def get_k_path_prob(model, tokenizer, query, k, max_new_tokens=80):
    logit = get_next_token_logit(model, tokenizer, query)
    k_token = logit[0].argsort()[-k:]
    k_prob = torch.nn.functional.softmax(logit[0][logit[0].argsort()[-k:]], dim=0)
    k_response = []
    for token in k_token:
        new_query = query + tokenizer.decode(token)
        candidate_inputs = tokenizer(new_query, return_tensors="pt")
        gen_out = model.generate(**candidate_inputs, output_scores=True, return_dict_in_generate=True, max_new_tokens=max_new_tokens)
        path_probs = get_path_prob(gen_out, k_prob)
        print(path_probs)
        print('----'*5)
        k_response.append(path_probs)
    return k_response

def get_follow_up_output(model, tokenizer, follow_up_template, gen_out, max_new_tokens=40):
    construct_input = lambda new_ids: {'input_ids': new_ids, "attention_mask":torch.ones_like(new_ids)}
    output_ids = gen_out.sequences
    follow_up_ids = tokenizer(follow_up_template, return_tensors="pt")['input_ids']
    new_ids = torch.concat([output_ids, follow_up_ids], axis=1)
    inputs = construct_input(new_ids)
    gen_out = model.generate(**inputs, output_scores=True, return_dict_in_generate=True, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    return gen_out

def get_k_path_prob_follow_up(model, tokenizer, query, k, max_new_tokens=80, 
                                follow_up_template=" So the answer is: "):
    logit = get_next_token_logit(model, tokenizer, query)
    k_token = logit[0].argsort()[-k:]
    k_prob = torch.nn.functional.softmax(logit[0][logit[0].argsort()[-k:]], dim=0)
    k_response = []
    for token in k_token:
        new_query = query + tokenizer.decode(token)
        candidate_inputs = tokenizer(new_query, return_tensors="pt")
        gen_out = model.generate(**candidate_inputs, output_scores=True, return_dict_in_generate=True, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
        
        follow_up_out = get_follow_up_output(model, tokenizer, follow_up_template, gen_out)
        path_probs = get_path_prob(follow_up_out, k_prob)

        print(path_probs)
        print('----'*5)
        k_response.append(path_probs)
    return k_response


