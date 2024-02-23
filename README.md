# CoT-decoding
Reimplementation of Chain-of-Thought Decoding

![image](https://github.com/fangyuan-ksgk/CoT-decoding/assets/66006349/f248d3f9-3b3b-4820-a20d-f6f1f9e38595)

Any HF model can be plugged-in and play

"""```python
query="A coin is heads up. Fletcher flips the coin. Conception flips the coin. Is the coin still heads up?"
template = """[INST]{question}[/INST]
               """
k_response = get_k_path_prob(model, tokenizer, template.format(question=query), k=5)
"""

