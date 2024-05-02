### Llama 2 7b Base 
With Alpaca-LoRA Fine-Tuning (PEFT method)
on prediction of essay understanding of PE,KE,LCE 

prompt:
   instruct = "Does the following essay explain the concept from the input correctly? Yes or no. " + curr_essay
   input = concept

### File explanations
1. llama_baseline_new.ipynb = zeroshot predictions
2. train_eval_kfold.py = fine-tuned model training and eval
3. eval_kfolds.py = eval code
4. train_eval_loss_charts.ipynb shows charts of train/eval loss

### HOW TO USE:
Create a new conda environment and install requirements: 
```
conda create -n env_name
conda activate env_name
pip install -r requirements.txt
```

```
python3 train_model.py --base_model='meta-llama/Llama-2-7b-hf' --output_dir='models/essay_model13' --num_epochs=65 --data_path='essay_train_examples2.json' --val_set_size=1 --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' --learning_rate=0.005 --cutoff_len=1000 
```


### Training/Finetuning Models
# PEFT parameters explained: 
   + --output_dir saves new fine-tuned model
   + --num_epochs = iterations model learns from data
   + --data_path = json file of instructions, 
   + --val_set_size = size of eval/test data
   + --lora_target_modules = loss function, each represents a linear layer
   + --learning_rate = step size to minimize loss 
   + --cutoff_len = cutoff for data

