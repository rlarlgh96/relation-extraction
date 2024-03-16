import pandas as pd
import numpy as np
import random
import torch
from transformers import AutoTokenizer, AutoConfig, Trainer, TrainingArguments

from model import CustomModel
from load_data import RE_Dataset, load_data, tokenized_dataset, label_to_num
from metrics import compute_metrics


def set_seed(seed:int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train():
  set_seed(42)

  # load model and tokenizer
  MODEL_NAME = "klue/roberta-large"
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  # add special tokens
  special_tokens = ['<S:ORG>','<S:PER>','<S:POH>','<S:LOC>','<S:DAT>','<S:NOH>','</S:ORG>','</S:PER>','</S:POH>','</S:LOC>','</S:DAT>','</S:NOH>','<O:ORG>','<O:PER>','<O:POH>','<O:LOC>','<O:DAT>','<O:NOH>','</O:ORG>','</O:PER>','</O:POH>','</O:LOC>','</O:DAT>','</O:NOH>']
  tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

  # load dataset
  train_dataset = load_data("../dataset/train/train_final.csv")
  validation_dataset = load_data("../dataset/validation/validation_final.csv")

  train_label = label_to_num(train_dataset['label'].values)
  validation_label = label_to_num(validation_dataset['label'].values)

  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  tokenized_validation = tokenized_dataset(validation_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  RE_validation_dataset = RE_Dataset(tokenized_validation, validation_label)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(device)

  # setting model hyperparameter
  model_config =  AutoConfig.from_pretrained(MODEL_NAME)
  model = CustomModel(MODEL_NAME, config=model_config)
  # resize token embeddings
  model.encoder.resize_token_embeddings(len(tokenizer))
  model.parameters
  model.to(device)

  training_args = TrainingArguments(
    output_dir='./results',          # output directory
    save_total_limit=5,              # number of total save model.
    save_steps=1000,                 # model saving step.
    num_train_epochs=10,              # total number of training epochs
    learning_rate=1e-5,               # learning_rate
    per_device_train_batch_size=64,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=1000,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_steps=1000,              # log saving step.
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                 # `no`: No evaluation during training.
                                 # `steps`: Evaluate every `eval_steps`.
                                 # `epoch`: Evaluate every end of epoch.
    eval_steps=1000,             # evaluation step.
    load_best_model_at_end=True,
    fp16=True
  )
  
  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=RE_train_dataset,
    eval_dataset=RE_validation_dataset,
    compute_metrics=compute_metrics
  )

  # train model
  trainer.train()
  save_path = f"./best_model/model.pth"
  torch.save(model.state_dict(), save_path)
  
def main():
    train()

if __name__ == '__main__':
  main()
  