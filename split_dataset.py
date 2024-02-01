"""
    DELETE THIS FILE later
"""
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split

#test_size = 75*(1/5)

data = pd.read_csv('essay_dataset.csv')
#split the data into train and test set
train,test = train_test_split(data, test_size=0.20, random_state=0)
#save the data
train.to_csv('train_data.csv',index=False)
test.to_csv('test_data.csv',index=False)
