import os

import torch
import transformers
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer


login(os.getenv("HUGGINGFACE_TOKEN"))

torch.set_grad_enabled(False)

# model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# input_string = "generate a value of a good AUC score on an arbitrary binary classification task. output nothing except the probability. you should generate no leading zero; the first character after 'answer: ' should be a decimal place '.'. answer: ."


def generate_probs(model_path, input_string):
    # messages = [
    #     {"role": "system", "content": ''},
    #     {"role": "user", "content": input_string},
    # ]

    # pipeline = transformers.pipeline(
    #     "text-generation",
    #     model=model_path,
    #     model_kwargs={"torch_dtype": torch.bfloat16},
    #     device="cuda",
    #     token=token,
    #     batch_size=8
    # )
    # pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id
    # outputs = pipeline(
    #     [messages],
    #     max_new_tokens=64,
    # )
    # results = [response[0]['generated_text'][-1]['content'] for response in outputs]

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_safetensors=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, use_safetensors=True)

    # tokenize to ids
    input_ids = tokenizer.encode(input_string, return_tensors="pt")

    # call model() to get logits
    logits = model(input_ids).logits

    # only care about the last projection in the last batch
    logits = logits[-1, -1]

    # softmax() to get probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # keep only the top 25
    probs, ids = torch.topk(probs, 25)

    # convert ids to tokens
    texts = tokenizer.convert_ids_to_tokens(ids)
    cleaned_tokens = [t.replace("Ċ", "").replace("Ġ", "") for t in texts]

    # print
    return list(zip(cleaned_tokens, probs))
    # for prob, text in zip(probs, cleaned_tokens):
    #     print(f"{prob:.4f}: \"{text}\"")
