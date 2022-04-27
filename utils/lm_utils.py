# The logic in these functions are adapted from the forward() implementations of various
# Causal LM models in the Transformers library. Each model has some variation on 
# the way logits and labels are preprocessed for the loss and it is not currently
# abstracted to a callable function.
import torch

def _preprocess_impl_XGLMForCausalLM(model, lm_logits, labels):
    # shift labels and add a pad token to the end
    shift_labels = labels.new_zeros(labels.shape)
    shift_labels[:, :-1] = labels[:, 1:].clone()
    shift_labels[:, -1] = model.config.pad_token_id
    return lm_logits, shift_labels    

def _preprocess_impl_GPT2LMHeadModel(model, lm_logits, labels):
    # Shift so that tokens < n predict n
    shift_logits = None if lm_logits is None else lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return shift_logits, shift_labels

def _preprocess_impl_GPTNeoForCausalLM(model, lm_logits, labels):
    # Compute loss in fp32 to match with mesh-tf version
    # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
    lm_logits = None if lm_logits is None else lm_logits.to(torch.float32)

    # Shift so that tokens < n predict n
    shift_logits = None if lm_logits is None else lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return shift_logits, shift_labels

def _preprocess_impl_GPTJForCausalLM(model, lm_logits, labels):
    # Shift so that tokens < n predict n
    shift_logits = None if lm_logits is None else lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return shift_logits, shift_labels

_preprocess_impl = {
    "XGLMForCausalLM": _preprocess_impl_XGLMForCausalLM,
    "GPT2LMHeadModel": _preprocess_impl_GPT2LMHeadModel,
    "GPTNeoForCausalLM": _preprocess_impl_GPTNeoForCausalLM,
    "GPTJForCausalLM": _preprocess_impl_GPTJForCausalLM
}

def preprocess_logits_and_labels_for_crossentropy(model, lm_logits, labels):
    model_type = type(model).__name__
    
    if model_type in _preprocess_impl:
        impl = _preprocess_impl[model_type]
        lm_logits, labels = impl(model, lm_logits, labels)
        
    return lm_logits, labels