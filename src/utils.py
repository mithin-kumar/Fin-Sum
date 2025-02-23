import transformers
import torch

from datasets import load_dataset

# print(datasets.__version__)
import transformers
import torch

from huggingface_hub import login

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
# import evaluate
import pandas as pd
import numpy as np


import yaml
import os
# if os.path.exists(os.getcwd)
print(os.getcwd())
with open('config\config.yaml','r') as f:
  config = yaml.safe_load(f)

with open('config\model_config.yaml','r') as f:
  model_config = yaml.safe_load(f)

with open('config\prompt_engineering.yaml','r') as f:
  prompt_config = yaml.safe_load(f)



print(config)
print(prompt_config)
def load_ds():
    print(config['dataset'])
    
    try:  
        ds = load_dataset(config['dataset'])
        print("dataset successfully installed")
        return ds
    
    except:
       print("unable to load")
       return "Loading dataset failed"

def load_model_tokenizer():
    model_name=model_config['MODEL_NAME']
    tokenizer_model = model_config['TOKENIZER_MODEL']
    original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    print("model and tokenizer load Successfull")
    return original_model, tokenizer

# load_model_tokenizer()
def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

# print(print_number_of_trainable_model_parameters(original_model))

def inference_model(saved_model):
    if os.path.exists(saved_model):
        instruct_model = AutoModelForSeq2SeqLM.from_pretrained(saved_model, torch_dtype=torch.bfloat16)
        tokenizer_model = model_config['TOKENIZER_MODEL']
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

        return instruct_model,tokenizer
    else:
        print(ImportError)
        return "failed to load model"
    
   
        

    