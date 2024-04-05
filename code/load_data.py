import pickle as pickle
import os
import pandas as pd
import torch


class RE_Dataset(torch.utils.data.Dataset):
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])

    return item

  def __len__(self):

    return len(self.labels)
  
def load_data(dataset_dir):
    dataset = pd.read_csv(dataset_dir)
    
    return dataset

def load_test_dataset(dataset_dir, tokenizer):
    test_dataset = load_data(dataset_dir)
    test_label = list(map(int,test_dataset['label'].values))
    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)

    return test_dataset['id'], tokenized_test, test_label

def tokenized_dataset(dataset, tokenizer):
  pairs, sentences = [], []
  ss, os = [], []

  for idx, row in dataset.iterrows():
    sentence = row['sentence']
    subject_word, object_word = row['subject_word'], row['object_word']
    subject_start_idx, subject_end_idx = row['subject_start_idx'], row['subject_end_idx']
    object_start_idx, object_end_idx = row['object_start_idx'], row['object_end_idx']
    subject_type, object_type = row['subject_type'], row['object_type']

    pair = subject_word + '와/과 ' + object_word +'의 관계'

    if subject_start_idx < object_start_idx:
      sentence = (sentence[:subject_start_idx] + 
                  f'<S:{subject_type}>' + 
                  sentence[subject_start_idx:subject_end_idx+1] + 
                  f'</S:{subject_type}>' + 
                  sentence[subject_end_idx+1:object_start_idx] + 
                  f'<O:{object_type}>' + 
                  sentence[object_start_idx:object_end_idx+1] + 
                  f'</O:{object_type}>' + 
                  sentence[object_end_idx+1:])
    else:
      sentence = (sentence[:object_start_idx] + 
                  f'<O:{object_type}>' + 
                  sentence[object_start_idx:object_end_idx+1] + 
                  f'</O:{object_type}>' + 
                  sentence[object_end_idx+1:subject_start_idx] + 
                  f'<S:{subject_type}>' + 
                  sentence[subject_start_idx:subject_end_idx+1] + 
                  f'</S:{subject_type}>' + 
                  sentence[subject_end_idx+1:])
      
    pairs.append(pair)
    sentences.append(sentence)

    encoded_inputs = tokenizer(
        sentence,
        truncation=True,
        max_length=180,
        add_special_tokens=True,
        )
    input_ids = encoded_inputs['input_ids']
    subject_token_id, object_token_id = tokenizer.convert_tokens_to_ids([f'<S:{subject_type}>', f'<O:{object_type}>'])
    
    if subject_token_id in input_ids:
      ss.append(input_ids.index(subject_token_id))
    else:
      ss.append(len(input_ids)-1)

    if object_token_id in input_ids:
      os.append(input_ids.index(object_token_id))
    else:
      os.append(len(input_ids)-1)

  tokenized_sentences = tokenizer( 
      pairs,
      sentences,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=180,
      add_special_tokens=True,
      )
  
  tokenized_sentences['ss'] = torch.tensor(ss, dtype=torch.long)
  tokenized_sentences['os'] = torch.tensor(os, dtype=torch.long)

  return tokenized_sentences

def label_to_num(label):
  num_label = []

  with open('./pickle/dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])

  return num_label

def num_to_label(label):
  origin_label = []

  with open('./pickle/dict_num_to_label.pkl', 'rb') as f:
    dict_num_to_label = pickle.load(f)
  for v in label:
    origin_label.append(dict_num_to_label[v])
    
  return origin_label
