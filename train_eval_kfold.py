import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
import pandas as pd
import gc

from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import Dataset
from eval_kfolds import eval_with_prompt
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score, classification_report

# Command line examples
# maximumn character len is 2200 for the essay dataset
# use model11: python3 train_model.py --base_model='meta-llama/Llama-2-7b-hf' --output_dir='models/essay_model11' --num_epochs=65 --data_path='essay_train_examples2.json' --val_set_size=1 --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' --learning_rate=0.005 --cutoff_len=1000 

# python3 train_model.py --base_model='meta-llama/Llama-2-7b-hf' --output_dir='models/essay_model13' --num_epochs=65 --data_path='essay_train_examples2.json' --val_set_size=1 --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' --learning_rate=0.005 --cutoff_len=1000 

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer
from utils.prompter import Prompter


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 2048,
    val_set_size: int = 0, # we are running our own cross validation
    # lora hyperparams
    lora_r: int = 4, # rank of the lora parameters. The smaller lora_r is, the fewer parameters lora has.
    lora_alpha: int = 8,
    lora_dropout: float = 0.05, # dropout probability for LoRA layers to prevent 
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj"
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if True, model learns to predict the input
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42 # fixed RNG for reproducibility 
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)
    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )
    
    # free memory 
    del model, tokenizer, trainer
    gc.collect() # for python memory management
    torch.cuda.empty_cache()  # for GPU memory 


label_mapper = {"PE":"Potential Energy", "KE":"Kinetic Energy", "LCE":"Law of Conservation of Energy"}

# Define the custom dataset for json creation
class EssayDatasetWithPrompt(Dataset):
    def __init__(self, essays, concept, labels):
        self.essays = essays
        self.concept = concept
        self.labels = labels

    def __len__(self):
        return len(self.essays)
    
    # returns training example in json string
    def write(self, file_name):
        with open(file_name, 'w') as fp:
            fp.write("[")
            num_row = 0
            for row_id, essay in enumerate(self.essays):
                label = self.labels[row_id].lower()
                if (label=='acceptable'):
                    label = "Yes"
                else:
                    label = "No"
                if (num_row > 0): #newline for 2nd example
                    fp.write( ",\n")
                fp.write("{\n")
                
                curr_essay = essay.replace("\t","")
                curr_essay = curr_essay.replace("\n","")
                # instruct = "Does the following essay explain the concept from the input correctly? Yes or no. " + input
                instruct = "You are given the following essay. \'" + curr_essay + \
                    "\' Does it explain the concept of \'" + label_mapper[self.concept] + \
                    "\'? Only answer yes or no."
                
                fp.write('"instruction": "' + instruct  + '",\n')
                fp.write('"input": "' + label_mapper[self.concept] + '",\n')
                # fp.write('"input": "",\n')

                fp.write('"output": "' + label + '"')
                fp.write("\n}")
                num_row = num_row + 1
            fp.write("]")

    
def fine_tune_with_prompt(concept, data_df):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # reproducibility

    # To store metrics for each fold
    fold_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": []
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(data_df, data_df[concept])):
        print(f"Training fold {fold + 1} for {concept}...")

        # Prepare datasets for the current fold
        train_dataset = EssayDatasetWithPrompt(data_df['Essay'].iloc[train_idx].values, concept, data_df[concept].iloc[train_idx].values)
        train_filename = "training_data/train_examples_" + concept + "_fold" + str(fold) + ".json"
        train_dataset.write(train_filename)
        eval_dataset = EssayDatasetWithPrompt(data_df['Essay'].iloc[val_idx].values, concept, data_df[concept].iloc[val_idx].values)

        # print(data_df['Essay'].iloc[train_idx].values)
        # print(eval_dataset)

        model_path = 'models/essay_model_' + concept + "_fold" + str(fold)
        # model_path = 'models/essay_model_PE_fold0'

        train(
            # model/data params
            base_model = 'meta-llama/Llama-2-7b-hf',  # the only required argument
            # base_model = 'meta-llama/Llama-2-7b-chat-hf',  # trying out chat
            data_path = train_filename,
            output_dir = model_path,
            # training hyperparams
            num_epochs = 50,
            learning_rate = 0.005,
            cutoff_len = 1024,
            val_set_size = 0,
            # lora hyperparams
            lora_target_modules = [ # linear layers 
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj"
            ]
        )
        
        # make predictions
        label_list, pred_list = eval_with_prompt(concept, eval_dataset, model_path)
        # exit(0)

        print(label_list)
        print(pred_list)
        # Calculate and store metrics
        accuracy = accuracy_score(label_list, pred_list)
        precision, recall, f1, _ = precision_recall_fscore_support(label_list, pred_list, average='weighted', zero_division=0)

        fold_metrics["accuracy"].append(accuracy)
        fold_metrics["precision"].append(precision)
        fold_metrics["recall"].append(recall)
        fold_metrics["f1"].append(f1)

        print("Metrics for PE:")
        print(f"Accuracy: {accuracy}, F1 Score: {f1}")
        print(f"Precision: {precision}, Recall: {recall}")

        # break # finish at 1 fold for now.

    # # Compute the average of the metrics across all folds
    avg_metrics = {metric: sum(values) / len(values) for metric, values in fold_metrics.items()}

    return avg_metrics    
    # exit(0)

def train_kfolds():
    # Standardize the labels and load data
    data_df = pd.read_excel("Student Essays Final Annotations.xlsx")
    data_df["PE"] = data_df["PE"].str.lower().str.strip()
    data_df["KE"] = data_df["KE"].str.lower().str.strip()
    data_df["LCE"] = data_df["LCE"].str.lower().str.strip()

    # Train the model separately for each concept
    metrics_PE = fine_tune_with_prompt("PE", data_df)
    metrics_KE = fine_tune_with_prompt("KE", data_df)
    metrics_LCE = fine_tune_with_prompt("LCE", data_df)

    print("Average metrics across all folds for PE:", metrics_PE)
    print("Average metrics across all folds for KE:", metrics_KE)
    print("Average metrics across all folds for LCE:", metrics_LCE)


if __name__ == "__main__":
    fire.Fire(train_kfolds)
