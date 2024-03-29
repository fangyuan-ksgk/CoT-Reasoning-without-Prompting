# Chain-of-Thought Reasoning Without Prompting
Unofficial of Chain-of-Thought Decoding @DeepMind

Great Work from Xuezhi Wang and Denny Zhou :-> Paper link: https://arxiv.org/abs/2402.10200

![image](https://github.com/fangyuan-ksgk/CoT-decoding/assets/66006349/f248d3f9-3b3b-4820-a20d-f6f1f9e38595)

Any HF model can be plugged-in and play

```python
from decode import *
query="A coin is heads up. Fletcher flips the coin. Conception flips the coin. Is the coin still heads up?"
template = """[INST]{question}[/INST]"""
k_response = get_k_path_prob(model, tokenizer, template.format(question=query), k=5)
```
k_response is a list of word and its probability like this:
```python
[('There', 0.005),
 ('is', 0.025),
 ('no', 0.965),
 ('certainty', 0.25),
 ('about', 0.015),
 ('the', 0.002),
 ('coin', 0.0),
 ('still', 0.0),
 ('being', 0.0),
 ('heads', 0.011),
 ('up', 0.001),
 ('after', 0.0),
 ('both', 0.092),
 ('flips', 0.072)]
```


On of the Issue with the original paper, is that it's difficult to locate the answer's position in the response. With a follow up question of "So the answer is", it is easier to parse the answer
```python
from decode import *
query="A coin is heads up. Fletcher flips the coin. Conception flips the coin. Is the coin still heads up?"
template = """[INST]{question}[/INST]"""
k_follow_up_response = get_k_path_prob_follow_up(model, tokenizer, template.format(question=query), k=5)
```


<img width="851" alt="Screenshot 2024-02-23 at 08 37 31" src="https://github.com/fangyuan-ksgk/CoT-Reasoning-without-Prompting/assets/66006349/d30dcb38-e207-429e-a8d6-2f13c62a3503">


