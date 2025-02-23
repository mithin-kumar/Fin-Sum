from utils import inference_model
import yaml
from transformers import  GenerationConfig
with open('config\prompt_engineering.yaml','r') as f:
    prompt_config = yaml.safe_load(f)
instruct_model , tokenizer = inference_model(saved_model='models\dialogue-summary-training-20250223T161311Z-001\dialogue-summary-training\checkpoint-10')

# print(model)
with open('dataset\\article.txt', 'r') as file:
    article = ''.join(file.readlines())





prompt = f"""
{prompt_config['start']}

{article}

{prompt_config['end']}
"""


print(prompt)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

instruct_model_outputs = instruct_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=2048, num_beams=1))
instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)

print(instruct_model_text_output)