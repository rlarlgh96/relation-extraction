import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader
import torch.nn.functional as F

from load_data import RE_Dataset, load_test_dataset, num_to_label
from model import CustomModel
from train import set_seed


def inference(model, tokenized_sent, device):
  dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
  model.eval()
  output_pred = []
  output_prob = []
  for i, data in enumerate(tqdm(dataloader)):
    with torch.no_grad():
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          ss = data['ss'].to(device),
          os = data['os'].to(device)
          )
    logits = outputs[0]
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
    output_prob.append(prob)

  return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def main():
  set_seed(42)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(device)

  # load tokenizer
  MODEL_NAME = "klue/roberta-large"
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  # add special tokens
  special_tokens = ['<S:ORG>','<S:PER>','<S:POH>','<S:LOC>','<S:DAT>','<S:NOH>','</S:ORG>','</S:PER>','</S:POH>','</S:LOC>','</S:DAT>','</S:NOH>','<O:ORG>','<O:PER>','<O:POH>','<O:LOC>','<O:DAT>','<O:NOH>','</O:ORG>','</O:PER>','</O:POH>','</O:LOC>','</O:DAT>','</O:NOH>']
  tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

  # load my model
  model_config =  AutoConfig.from_pretrained(MODEL_NAME)
  model = CustomModel(MODEL_NAME, config=model_config)
  # resize token embeddings
  model.encoder.resize_token_embeddings(len(tokenizer))
  model.load_state_dict(torch.load("./best_model/model.pth"))
  model.parameters
  model.to(device)

  # load test datset
  test_dataset_dir = "../dataset/test/test_final.csv"
  test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
  RE_test_dataset = RE_Dataset(test_dataset ,test_label)

  # predict answer
  pred_answer, output_prob = inference(model, RE_test_dataset, device)
  pred_answer = num_to_label(pred_answer)
  
  # make csv file with predicted answer
  output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})

  output.to_csv('./prediction/submission.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
  print('---- Finish! ----')

if __name__ == '__main__':
  main()