from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import torch

model_name = "HuggingFaceH4/zephyr-7b-beta"

def top_p_logits(logits, topp=0.9, filter_value=0, min_topk=1):
    cum_logits = logits.clone()
    if topp > 0:
        logits_sorted, inds = torch.sort(logits, dim=-1, descending=True)
        mask = (logits_sorted.cumsum(dim=-1) - logits_sorted) >= topp
        mask[:, :min_topk] = False
        # Remove tokens with cumulative top_p above the threshold
        mask = torch.zeros_like(mask).to(torch.bool).scatter_(dim=-1, index=inds, src=mask)
        cum_logits[mask] = filter_value
        cum_logits.div_(cum_logits.sum(dim=-1, keepdim=True))

    return cum_logits

class ContrastiveDecodingModel:
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @torch.no_grad()
    def contra_generate(self, input_within, input_without, **kwargs):
        """
        Generate response
        """
        maxlen_res = kwargs.pop('max_new_tokens', 48)
        temperature = kwargs.pop('temperature', 0.7)
        topp = kwargs.pop('topp', 0.8)
        ratio = kwargs.pop('ratio', 2)

        dev = input_within.device
        bsz = 1

        done = torch.zeros((bsz,), device=dev).to(torch.bool)

        inds = torch.arange(bsz).to(dev)
        input_within = torch.index_select(input_within, 0, inds)
        input_without = torch.index_select(input_without, 0, inds)

        init_length_in = input_within.size(1)
        init_length_out = input_without.size(1)

        def score_process(score, input_within, input_without):
            score = score[:, -1, :]

            score = torch.softmax(score.div(temperature), dim=-1)
            probs = top_p_logits(score, topp=topp, filter_value=0)
            tok_ids = torch.argmax(probs, dim=-1).to(input_within.device)
            hyp_ids = torch.arange(probs.size(0), device=dev)

            tok_ids = torch.where(done, self.tokenizer.pad_token_id, tok_ids)
            input_within = torch.cat((input_within, tok_ids.unsqueeze(-1)), dim=-1)
            input_without = torch.cat((input_without, tok_ids.unsqueeze(-1)), dim=-1)

            return input_within, input_without, tok_ids, hyp_ids

        for _token in range(maxlen_res):
            if done.all():
                break
            
            score_in = self.model(input_within)[0]
            score_out = self.model(input_without)[0]

            score_in[:, -1, :] = score_in[:, -1, :] + ratio * (score_in[:, -1, :] - score_out[:, -1, :])

            input_within, input_without, tok_ids, hyp_ids = score_process(score_in, input_within, input_without)

            done = done | tok_ids.eq(self.tokenizer.eos_token_id)

        input_within = input_within[:, init_length_in:]
        input_within = input_within.view(bsz, -1)
        input_without = input_without[:, init_length_out:]
        input_without = input_without.view(bsz, -1)

        return input_within, input_without