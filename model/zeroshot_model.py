from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
#import torch

#from utils.tokenize_helpers import tokenize_with_left_truncation

class CSN_Zeroshot(nn.Module):
    def __init__(self, model_path):
        super(CSN_Zeroshot, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if not self.tokenizer._pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        #self.model_max_length = self.model.config.max_position_embeddings
        
        self.yes_token_ids = self.tokenizer.convert_tokens_to_ids(["对", "是", "正"])
        self.no_token_ids = self.tokenizer.convert_tokens_to_ids(["错", "否", "不"])
        #self.yes_token_idx = self.tokenizer.convert_tokens_to_ids("yes")
        #self.no_token_idx = self.tokenizer.convert_tokens_to_ids("no")
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, CSSs, sent_char_lens, mention_poses, quote_idxes, true_index, device):
        candidate_aliases = [CSS[cdd_pos[1]:cdd_pos[2]] for cdd_pos, CSS in zip(mention_poses, CSSs)]
        quoted_sentences = []
        for i, (cdd_CSS, cdd_sent_char_lens, cdd_quote_idx) in enumerate(zip(CSSs, sent_char_lens, quote_idxes)):
            accum_char_len = [0]
            for sent_idx in range(len(cdd_sent_char_lens)):
                accum_char_len.append(accum_char_len[-1] + cdd_sent_char_lens[sent_idx])
            quoted_sentences.append(cdd_CSS[accum_char_len[cdd_quote_idx]:accum_char_len[cdd_quote_idx + 1]])
        
        #prompts = [f"Text:'{css}'\n\nQuestion: yes or no: did {alias} say {quoted}?\n\nAnswer:"
        #prompts = [f"文字：'{css}'\n\n问题：那个说{quoted}是{alias}。对或错？\n\n回答："
        prompts = [f"文字：'{css}'\n\n问题：那个说{quoted}是{alias}。对或错？\n\n答案是"
                   for css, alias, quoted in zip(CSSs, candidate_aliases, quoted_sentences)]
        prompts = [p.replace("“", '"').replace("”", '"') for p in prompts]
        inputs = self.tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors="pt").to(device)
        #inputs = tokenize_with_left_truncation(self.tokenizer, prompts, max_length=self.model_max_length, 
        #                                       return_tensors="pt").to(device)
        #decoder_input_ids = torch.tensor([[self.tokenizer.pad_token_id]]*len(prompts), dtype=torch.long).to(device)
        #logits = self.model(**inputs, decoder_input_ids=decoder_input_ids).logits
        logits = self.model(**inputs).logits
        yes_no_logits = logits[:, -1, self.yes_token_ids + self.no_token_ids]
        yes_no_probs = self.softmax(yes_no_logits)
        
        yes_probs = yes_no_probs[:, :len(self.yes_token_ids)].sum(dim=-1)
        no_probs = yes_no_probs[:, -len(self.no_token_ids):].sum(dim=-1)
        scores = yes_probs - no_probs
        scores_false = [scores[i] for i in range(scores.size(0)) if i != true_index]
        scores_true = [scores[true_index] for i in range(scores.size(0) - 1)]
        
        return scores, scores_false, scores_true