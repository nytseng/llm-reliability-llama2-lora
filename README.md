# Llama 2 7b Base Model
With Alpaca-LoRA Fine-Tuning (PEFT method)
on prediction of essay understanding of PE,KE,LCE 

### STATUS REPORT: working on prompt engineering for better results.
## current best model is 50 epochs, learning rate 0.005, train_on_input = False?
prompt:
   instruct = "Does the following essay explain the concept from the input correctly? Yes or no. " + curr_essay
   input = concept

### File explanations
1. essay_predict_baseline.py = zeroshot predictions
2. train_eval_kfold.py = fine-tuned model training and eval
3. eval_kfolds.py = eval code

### Training/Finetuning Models
# parameters explained: 
   + --output_dir saves new fine-tuned model
   + --num_epochs = iterations model learns from data
   + --data_path = json file of instructions, 
   + --val_set_size = size of eval/test data
   + --lora_target_modules = loss function, each represents a linear layer
   + --learning_rate = step size to minimize loss 
   + --cutoff_len = cutoff for data

### TODO priority: 
1. Fine-Tuning aim to reduce training loss under 0.1
   - If training loss continues to decline with 25 epochs, increase epochs. 
   - if training loss fluctuates up and down (on wandb), reduce learning rate.
3. Prompt engineering, (make sure that training instruction is same as testing)

### TODO: 
5. clean up code, comment documentation
6. move code into jupyter notebook
