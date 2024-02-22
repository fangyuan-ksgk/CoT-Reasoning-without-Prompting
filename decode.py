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

# model_name = "mistralai/Mistral-7B-v0.1"
model_name = "mlabonne/Monarch-7B"
# model_name = "mistralai/Mixtral-8x7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

""""
--- Cool Function ---
* Check the log probability of any response given a query | Given a huggingface Model & Tokenizer 
"""
def check_response_logprob(model, tokenizer, query, target_response):
    inputs = tokenizer([query], return_tensors="pt")
    gen_out = model.generate(**inputs, output_scores=True, return_dict_in_generate=True)

    target_ids = tokenizer.encode(target_response)
    sum_of_logits = 0
    for i, id in enumerate(target_ids):
        sum_of_logits += gen_out.scores[i][0, id]

    return sum_of_logits


"""
Now, I wish to do tree-search to locate the confident answer, and NOT the confident continuation
1. Should be doable by checking on the next-token
"""
def get_next_token_logit(model, tokenizer, query):
    inputs = tokenizer([query], return_tensors="pt")
    gen_out = model.generate(**inputs, output_scores=True, return_dict_in_generate=True, max_new_tokens=1)
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
def get_token_path_prob(gen_out):
    logits = gen_out.scores
    num_output = len(logits)
    output_ids = gen_out.sequences[0][-num_output+1:]
    # output = tokenizer.decode(output_ids, skip_special_tokens=True)
    path_prob = torch.stack([score[0].max() for score in logits])
    path_prob = torch.nn.functional.softmax(path_prob)
    # path_logprob = torch.log(path_prob)
    return output_ids, path_prob
    
# Word Path Probability -- Ensemble(word[token1,token2,...]) is the average probability of token appearance likelihood
def get_path_prob(gen_out, init_token_prob):
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
    k_prob = torch.nn.functional.softmax(logit[0][logit[0].argsort()[-k:]])
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

# 
def get_log_prob(logits, token_id):
    # Compute the softmax of the logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    log_probabilities = torch.log(probabilities)
    
    # Get the log probability of the token
    token_log_probability = log_probabilities[token_id].item()
    return token_log_probability

def greedy_search(input_ids, node, graph, length=5):
    if length == 0:
        return input_ids

    outputs = model(input_ids)
    predictions = outputs.logits

    # Get the predicted next sub-word (here we use top-k search)
    logits = predictions[0, -1, :]
    token_id = torch.argmax(logits).unsqueeze(0)

    # Compute the score of the predicted token
    token_score = get_log_prob(logits, token_id)

    # Add the predicted token to the list of input ids
    new_input_ids = torch.cat([input_ids, token_id.unsqueeze(0)], dim=-1)

    # Add node and edge to graph
    next_token = tokenizer.decode(token_id, skip_special_tokens=True)
    current_node = list(graph.successors(node))[0]
    graph.nodes[current_node]['tokenscore'] = np.exp(token_score) * 100
    graph.nodes[current_node]['token'] = next_token + f"_{length}"

    # Recursive call
    input_ids = greedy_search(new_input_ids, current_node, length-1)
    
    return input_ids


def plot_graph(graph, length, beams, score):
    fig, ax = plt.subplots(figsize=(3+1.2*beams**length, max(5, 2+length)), dpi=300, facecolor='white')

    # Create positions for each node
    pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")

    # Normalize the colors along the range of token scores
    if score == 'token':
        scores = [data['tokenscore'] for _, data in graph.nodes(data=True) if data['token'] is not None]
    elif score == 'sequence':
        scores = [data['sequencescore'] for _, data in graph.nodes(data=True) if data['token'] is not None]
    vmin = min(scores)
    vmax = max(scores)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list('rg', ["r", "y", "g"], N=256) 

    # Draw the nodes
    nx.draw_networkx_nodes(graph, pos, node_size=2000, node_shape='o', alpha=1, linewidths=4, 
                          node_color=scores, cmap=cmap)

    # Draw the edges
    nx.draw_networkx_edges(graph, pos)

    # Draw the labels
    if score == 'token':
        labels = {node: data['token'].split('_')[0] + f"\n{data['tokenscore']:.2f}%" for node, data in graph.nodes(data=True) if data['token'] is not None}
    elif score == 'sequence':
        labels = {node: data['token'].split('_')[0] + f"\n{data['sequencescore']:.2f}" for node, data in graph.nodes(data=True) if data['token'] is not None}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10)
    plt.box(False)

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    if score == 'token':
        fig.colorbar(sm, ax=ax, orientation='vertical', pad=0, label='Token probability (%)')
    elif score == 'sequence':
        fig.colorbar(sm, ax=ax, orientation='vertical', pad=0, label='Sequence score')
    plt.show()



def get_best_sequence(G):
    # Create a list of leaf nodes
    leaf_nodes = [node for node in G.nodes() if G.out_degree(node)==0]

    # Get the leaf node with the highest cumscore
    max_score_node = None
    max_score = float('-inf')
    for node in leaf_nodes:
        if G.nodes[node]['sequencescore'] > max_score:
            max_score = G.nodes[node]['sequencescore']
            max_score_node = node

    # Retrieve the sequence of nodes from this leaf node to the root node in a list
    path = nx.shortest_path(G, source=0, target=max_score_node)

    # Return the string of token attributes of this sequence
    sequence = "".join([G.nodes[node]['token'].split('_')[0] for node in path])
    
    return sequence, max_score