"""
Creates json file containing instructions for the training data
"""
from datasets import load_dataset
import json
import pandas

dataset = pandas.read_csv('train_data.csv') # replace csv with training data
labels = ["Potential Energy", "Kinetic Energy", "Law of Conservation of Energy"]
label_col_names = ["PE", "KE", "LCE"]
input_examples = dataset["Essay"]
# with open('train_data1.json', 'w') as fp:
#     fp.write("[\n")
#     fp.close()

for i, label in enumerate(labels):
    with open('train_data_'+label_col_names[i]+'.json', 'w') as fp:
        fp.write("[")
        num_row = 0
        for row in range(len(dataset)):
            if (num_row > 0): #newline for 2nd example
                fp.write( ",\n")
            fp.write("{\n")
            input = dataset["Essay"][row]
            input = input.replace("\n","")
            input = input.replace("\t","")
            instruct = "Does the following essay explain the concept from the input correctly? Yes or no. " + input
            fp.write('"instruction": "' + instruct  + '",\n')
            
            fp.write('"input": "' + label + '",\n')
            output = dataset[label_col_names[i]][row]
            if output.lower() == "acceptable":
                output = "Yes"
            else:
                output = "No"
            fp.write('"output": "' + output + '"')
            fp.write("\n}")
            num_row = num_row + 1
        fp.write("]")
