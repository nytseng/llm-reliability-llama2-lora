# Llama 2 7b Base Model
# With Alpaca-LoRA Fine-Tuning (PEFT method)
# on prediction of essay understanding of PE,KE,LCE 

### STATUS REPORT: Need to implement same 5fold CV as other models (reproducibility), 
## continue fine-tuning, create separate models for CU0, CU1, ... CU5.

### File explanations
1. essay_predict_baseline.py = zeroshot predictions
2. split_dataset.py = split training and testing data to csv for reproducibility and testing.
3. dataset_to_train_json.py = turns training data into json of instructions
3. train_model.py = fine-tuned model training
4. test_model.py = testing
5. f1_score_baseline/finetuned.txt = calculate metrics

### Fine-Tuning Walkthrough:
1. split_dataset.py (train test split)
2. dataset_to_train_json.py (puts training data into instructions to run)
3. train_model.py (fine tuning models)
4. test_model.py  (evaluate on the test data)

### TODO priority: 
1. Set up reproducible 5Fold Cross Validation
2. Fine-Tuning (learning rate, epoch) that reduces training loss under 0.1
   - If training loss continues to decline with 25 epochs, increase epochs. 
   - if training loss fluctuates up and down (check on wandb), reduce learning rate.
3. Prompt engineering, make sure that training instruction is same as testing


### TODO priority: 
4. Set up git repository, makes things easier
5. clean up code, comment documentation
6. move code into jupyter notebook


### Training/Finetuning Models
# sample command in terminal:
python3 train_model.py --base_model='meta-llama/Llama-2-7b-hf' --output_dir='models/essay_model14' --num_epochs=65 --data_path='train_data2.json' --val_set_size=1 --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' --learning_rate=0.005 --cutoff_len=2048
# parameters explained: 
   + --output_dir saves new fine-tuned model
   + --num_epochs = iterations model learns from data
   + --data_path = json file of instructions, 
   + --val_set_size = size of eval/test data
   + --lora_target_modules = loss function, each represents a linear layer
   + --learning_rate = step size to minimize loss 
   + --cutoff_len = cutoff for data


##### Fine-tuning ITERATIONS
## 1/29 train with 65 epochs, adjust learning rate: 
# 65 epochs may be too overfitting
python3 train_model.py --base_model='meta-llama/Llama-2-7b-hf' --output_dir='models/essay_model14' --num_epochs=65 --data_path='train_data2.json' --val_set_size=1 --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' --learning_rate=0.005 --cutoff_len=1000 
; python3 train_model.py --base_model='meta-llama/Llama-2-7b-hf' --output_dir='models/essay_model15' --num_epochs=65 --data_path='train_data2.json' --val_set_size=1 --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' --learning_rate=0.004 --cutoff_len=1000 
; python3 train_model.py --base_model='meta-llama/Llama-2-7b-hf' --output_dir='models/essay_model16' --num_epochs=65 --data_path='train_data2.json' --val_set_size=1 --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' --learning_rate=0.006 --cutoff_len=1000 

## after 50 epochs, training loss is stable