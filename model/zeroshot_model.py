from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch.nn as nn
import torch

from utils.lm_utils import preprocess_logits_and_labels_for_crossentropy

class CSN_Zeroshot(nn.Module):
    def __init__(self, args):
        super(CSN_Zeroshot, self).__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_pretrained_dir)
        if not self.tokenizer._pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        config = AutoConfig.from_pretrained(args.bert_pretrained_dir)
        if config.is_encoder_decoder:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(args.bert_pretrained_dir)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(args.bert_pretrained_dir)
            
        self.loss_fct = nn.CrossEntropyLoss(reduction="none")
        
        if args.prompt_lang == "zh":
            self.prompt = "文字：'{css}'\n\n问题：那个说{quoted}是{alias}。对或错？\n\n"
            self.prompt += "答案是" if args.prompt_style == "answer_is" else "回答："
            self.response_yes = "是的"
            self.response_no = "不对"
            self.space = ""
        else:
            self.prompt = "Text:'{css}'\n\nQuestion: The one who said {quoted} is {alias}. True or false?\n\n"
            self.prompt += "The answer is" if args.prompt_style == "answer_is" else "Answer:"
            self.response_yes = "true"
            self.response_no = "false"
            self.space = " "
            
    def forward(self, CSSs, sent_char_lens, mention_poses, quote_idxes, true_index, device):
        candidate_aliases = [CSS[cdd_pos[1]:cdd_pos[2]] for cdd_pos, CSS in zip(mention_poses, CSSs)]
        quoted_sentences = []
        for i, (cdd_CSS, cdd_sent_char_lens, cdd_quote_idx) in enumerate(zip(CSSs, sent_char_lens, quote_idxes)):
            accum_char_len = [0]
            for sent_idx in range(len(cdd_sent_char_lens)):
                accum_char_len.append(accum_char_len[-1] + cdd_sent_char_lens[sent_idx])
            quoted_sentences.append(cdd_CSS[accum_char_len[cdd_quote_idx]:accum_char_len[cdd_quote_idx + 1]])
        
        prompts = [self.prompt.format(css=css, alias=alias, quoted=quoted)
                   for css, alias, quoted in zip(CSSs, candidate_aliases, quoted_sentences)]
        prompts = [p.replace("“", '"').replace("”", '"') for p in prompts]
        
        yes_ppl = self.get_ppl(prompts, self.response_yes, device)
        no_ppl = self.get_ppl(prompts, self.response_no, device)
        
        scores = (no_ppl - yes_ppl) / (no_ppl + yes_ppl)
        scores_false = [scores[i] for i in range(scores.size(0)) if i != true_index]
        scores_true = [scores[true_index] for i in range(scores.size(0) - 1)]
        
        return scores, scores_false, scores_true
    
    def get_ppl(self, prompts, response, device):
        base_targets = None
        if self.model.config.is_encoder_decoder:
            inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(device)
            responses = [response] * len(prompts)
            with self.tokenizer.as_target_tokenizer():
                targets = self.tokenizer(responses, padding=True, return_tensors="pt").to(device)
                
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(targets.input_ids)
            logits = self.model(**inputs, decoder_input_ids=decoder_input_ids, 
                                decoder_attention_mask=targets.attention_mask).logits
        else:
            prompts_with_response = [f"{p}{self.space}{response}" for p in prompts]
            inputs = self.tokenizer(prompts_with_response, padding=True, return_tensors="pt").to(device)
            base_targets = self.tokenizer(prompts, padding=True, return_tensors="pt").to(device)
            targets = inputs
            logits = self.model(**inputs).logits
        
        ppl = torch.exp(self._get_lm_loss(logits, targets, base_targets))
        return ppl
    
    def _get_lm_loss(self, logits, targets, base_targets):
        labels = targets.input_ids
        base_labels = None if base_targets is None else base_targets.input_ids
        
        # Get the token-wise loss
        logits, labels = preprocess_logits_and_labels_for_crossentropy(self.model, logits, labels)
        logits = torch.permute(logits, (0, 2, 1))
        loss = self.loss_fct(logits, labels)
        
        # Mask out any position in labels that is identical to base_labels
        # and also mask out any padding
        mask = torch.ones_like(loss)
        if base_labels is not None:
            _, base_labels = preprocess_logits_and_labels_for_crossentropy(self.model, None, base_labels)
            mask[:, :base_labels.shape[1]] = labels[:, :base_labels.shape[1]] != base_labels
        mask[labels == self.tokenizer.pad_token_id] = 0.
        loss = (loss*mask).sum(dim=-1) / mask.sum(dim=-1)
        return loss
        