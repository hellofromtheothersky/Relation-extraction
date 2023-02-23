# Import
import numpy as np
import pandas as pd
import pickle

from tqdm import tqdm
from transformers import BertTokenizer
from tensorflow.data import Dataset

# Data needed
with open('data/data_encoder.obj', 'rb') as f:
    Encoder=pickle.load(f)
max_len= Encoder.max_len

# Preprocessing data
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

def map_dataset(input_ids, attn_masks, e1_position, e2_position, grammar_relation, shortest_path, labels):
    return {
        'input_ids': input_ids,
        'attention_mask': attn_masks,
        'e1_position': e1_position,
        'e2_position': e2_position,
        'grammar_relation': grammar_relation,
        'shortest_path': shortest_path
    }, labels

def empty_matrix(data, max_len=max_len):
    return np.zeros((len(data), max_len))

# Class for preprocessing data
class preprocess_data_BERT():
    def __init__(self, data, X, y):
        self.data = data
        self.X = X
        self.y = y
    
    def generate_data(self, ids, masks, tokenizer=tokenizer):
        for i, text in tqdm(enumerate(self.data['sentence'])):
            tokenized_text = tokenizer.encode_plus(text,
                                                   max_length=max_len,
                                                   truncation=True,
                                                   padding='max_length',
                                                   add_special_tokens=True,
                                                   return_tensors='tf')
            ids[i, :] = tokenized_text.input_ids
            masks[i, :] = tokenized_text.attention_mask
        ids_data, attn_data = empty_matrix(self.data), empty_matrix(self.data)
        ids_data, attn_data = ids, masks
        dataset = Dataset.from_tensor_slices((ids_data, attn_data, self.X[1], self.X[2], self.X[3], self.X[4], self.y))
        dataset = dataset.map(map_dataset)
        dataset = dataset.shuffle(100).batch(16, drop_remainder=True)
        return dataset

