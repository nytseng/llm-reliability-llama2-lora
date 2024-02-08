import torch
from peft import PeftModel
import transformers
import gradio as gr
import re
import pandas
import csv
import gc

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

BASE_MODEL = "meta-llama/Llama-2-7b-hf"
LORA_WEIGHTS = "models/essay_model14"

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

model = None

# if device == "cuda":
#     model = LlamaForCausalLM.from_pretrained(
#         BASE_MODEL,
#         load_in_8bit=False,
#         torch_dtype=torch.float16,
#         device_map="auto",
#     )
#     model = PeftModel.from_pretrained(
#         model, LORA_WEIGHTS, torch_dtype=torch.float16, force_download=True, 
#     )
# elif device == "mps":
#     model = LlamaForCausalLM.from_pretrained(
#         BASE_MODEL,
#         device_map={"": device},
#         torch_dtype=torch.float16,
#     )
#     model = PeftModel.from_pretrained(
#         model,
#         LORA_WEIGHTS,
#         device_map={"": device},
#         torch_dtype=torch.float16,
#     )
# else:
#     model = LlamaForCausalLM.from_pretrained(
#         BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
#     )
#     model = PeftModel.from_pretrained(
#         model,
#         LORA_WEIGHTS,
#         device_map={"": device},
#     )

def remove_unwanted_symbols(s):
    new_column = ""
    for c in s:
        if c.isalnum() or c.isspace() or c=='_' or c==',' or c=='=' or c=="." \
                or c=='<' or c=='>' or c=="'" or c=="(" or c==")" or c=="*" or c=="!" \
                or c=='-' or c=='%' or c=='/' or c=='[' or c==']':
            new_column += c
        else:
            #print("stop")
            #print(c)
            break
    return new_column

def translate(s):
    column = s
    # trimming unneeded content
    if '+' in column:
        column = column.split('+')[0]
    # if '⁇' in column:
    #     column = column.split('⁇')[0]
    column = remove_unwanted_symbols(column)

    #direct replacement
    column = column.replace("trim_vendor_item","TrimVenItem")
    column = column.replace("trim_vendor_item","TrimVenItem")
    column = column.replace("source_vendor_item", "SrcVenItem")
    column = column.replace("medline_suggested_substitute", "Medline Suggested Sub")
    column = column.replace("parent_monthly_inventory_on_hand", "Parent MIOH")
    column = column.replace("invoice_paid_total", "InvcPaidTotal")
    # table rename
    column = column.replace("Purchase_Order_System", "POS")
    column = column.replace("Potential_Back_Order","PBO_Sam_PVA")
    column = column.replace("Supply_Items_At_Risk", "SupplyItemsAtRisk")
    column = column.replace("Purchase_Order_Receipts", "Receipts")

    #indirect replacement
    column = column.replace("source","src")
    column = column.replace("purchase_order","PO")
    column = column.replace("account","acct")
    column = column.replace("description","desc")
    column = column.replace("vendor", "vend")
    column = column.replace("transaction", "trx")
    column = column.replace("quantity","qty")
    column = column.replace("manufacturer","manuf")
    column = column.replace("process","proc")
    column = column.replace("recommended","rec")
    column = column.replace("location", "loc")
    column = column.replace("request", "Req")
    column = column.replace("monthly_inventory_on_hand", "MIOH")
    column = column.replace("substitute", "sub")
    column = column.replace("srcReqer", "SrcRequester")
    if "FROM PBO_Sam_PVA" in column:
        column = column.replace("_"," ")
    else:
        column = column.replace("_","")
    return column


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


def evaluate(
    instruction,
    input=None,
    temperature=1.0,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=4, # max output tokens, only needs yes/no. this speeds up runtime
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

label_mapper = {"PE":"Potential Energy", "KE":"Kinetic Energy", "LCE":"Law of Conservation of Energy"}

def eval_with_prompt(concept, essay_struct, model_path):
    # load fine-tuned model
    global model 
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model, model_path, torch_dtype=torch.float16, force_download=True, 
    )

    if device != "cpu":
        model.half()
    model.eval()
    if torch.__version__ >= "2":
        model = torch.compile(model)

    # print(essay_struct.essays)
    
    pred_list = [] # predictions
    label_list = []
    for row_id, essay in enumerate(essay_struct.essays):
        label = essay_struct.labels[row_id].lower()
        if (label=='acceptable'):
            expected = "Yes"
        else:
            expected = "No"
            label = "unacceptable"
        
        curr_essay = essay.replace("\t","")
        curr_essay = curr_essay.replace("\n","")
        ### NOTE: make sure that training instruction is same as testing instr
        # instruct = "Does the following essay explain the concept from the input correctly? Yes or no. " + curr_essay # input here is essay
        
        instruct = "You are a junior high school teacher grading the answers for a Physics " + \
                     "test. You are given a paragraph written by a student to describe a physics concept. " + \
                     "Please say Yes if it defines the concept at the input correctly, otherwise say No. " + \
                     "Here is the student paragraph: \'" + curr_essay + "\'"  
       
        output = evaluate(instruction = instruct, input = label_mapper[concept])
        # output = evaluate(instruction = instruct, input = "")
        # print("instruct = " + instruct)
        # print("label_mapper[concept] = " + label_mapper[concept])
        # exit()
        print("Concept:" + concept + ", Expected:" + expected + ", Model Response:" + output)

        label_list.append(label)
        if "yes" in output.lower() and len(output.split()) < 5: # in case that the output contains "yes" from input
            pred_list.append("acceptable")
            print("Model Response: yes")

        else: 
            pred_list.append("unacceptable")
            print("Model Response: no")


    # free memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return label_list, pred_list


