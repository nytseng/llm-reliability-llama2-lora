import torch
from peft import PeftModel
import transformers
import pandas as pd
import csv

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

#BASE_MODEL = "decapoda-research/llama-7b-hf"
BASE_MODEL = "meta-llama/Llama-2-7b-hf"
LORA_WEIGHTS = "models/l2model8"

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    # model = PeftModel.from_pretrained(
    #     model, LORA_WEIGHTS, torch_dtype=torch.float16, force_download=True, 
    # )
elif device == "mps":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
else:
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
    )


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{input}
### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response:"""

if device != "cpu":
    model.half()
model.eval()
if torch.__version__ >= "2":
    model = torch.compile(model)


def evaluate(
    instruction,
    input=None,
    temperature=1.0,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=15,
    **kwargs,
):
    with torch.autocast("cuda"):
        prompt = generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt") 
        # inputs = tokenizer(prompt, max_length=2048 , return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return output.split("### Response:")[1].strip()


def evaluate_prompt_and_instruction(instruction, input):
    return evaluate(instruction, input)


if __name__ == "__main__":
    csvFile = pd.read_csv('essay_dataset.csv')
    labels = ["Potential Energy", "Kinetic Energy", "Law of Conservation of Energy"]
    input_examples = csvFile["Essay"]
    with open('predict_output_baseline.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["Essay ID","Essay","PE_Predicted","KE_Predicted","LCE_Predicted", "PE_Actual","KE_Actual","LCE_Actual"]
        writer.writerow(field)

        for row in range(len(input_examples)):
            input = input_examples[row]
            predicted = ["unacceptable","unacceptable","unacceptable"]
            
            for i, label in enumerate(labels):
                instruct = "Does the following essay explain the concept from the input correctly? Yes or no. " + input
                s = evaluate(instruction = instruct, input = label)
                
                print("Expected:" + label + ",  Response:", s)
                
                if "yes" in s.lower():
                    predicted[i] = "acceptable"

            writer.writerow([csvFile["Essay ID"][row], input, predicted[0], predicted[1], predicted[2], csvFile["PE"][row],csvFile["KE"][row],csvFile["LCE"][row]])
        
