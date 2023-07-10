import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import datasets
import numpy as np
import pandas as pd
import math
import csv
import sys
import torch
from transformers import BertTokenizerFast
from transformers import BertForTokenClassification
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from torch.utils.data import DataLoader
from torch.optim import SGD
from tqdm import tqdm

testfile = sys.argv[1]
outputfile = sys.argv[2]

def parse_csv(filename):
  sentencelist = []
  words = []
  with open(filename) as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
      if row:
        words.append(row[1:])
      else:
        if words:
          sentencelist.append(words)
          words = []
    if words:
      sentencelist.append(words)
  return sentencelist

def parse_csv_test(filename):
  sentencelist = []
  words = ""
  with open(filename) as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
      if row:
        words += row[1] + " "
      else:
        if words:
          words = words[:-1]
          sentencelist.append(words)
          words = ""
    if words:
      sentencelist.append(words)
  return sentencelist

label_all_tokens = False
def align_label(texts, labels):
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids

class DataSequence(torch.utils.data.Dataset):

    def __init__(self, df):

        lb = df['labels']
        txt = df['text']
        self.texts = [tokenizer(str(i),
                               padding='max_length', max_length = 512, truncation=True, return_tensors="pt") for i in txt]
        self.labels = [align_label(i,j) for i,j in zip(txt, lb)]

    def __len__(self):

        return len(self.labels)

    def get_batch_data(self, idx):

        return self.texts[idx]

    def get_batch_labels(self, idx):

        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):

        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)

        return batch_data, batch_labels
    
class BertModel(torch.nn.Module):

    def __init__(self):

        super(BertModel, self).__init__()

        self.bert = BertForTokenClassification.from_pretrained('Twitter/twhin-bert-base', num_labels=len(unique_labels))

    def forward(self, input_id, mask, label):

        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)

        return output
    

def train_loop(model, df_train, df_val):

    train_dataset = DataSequence(df_train)
    val_dataset = DataSequence(df_val)

    train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

    if use_cuda:
        model = model.cuda()

    best_acc = 0
    best_loss = 1000

    for epoch_num in range(EPOCHS):

        total_acc_train = 0
        total_loss_train = 0

        model.train()

        for train_data, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            mask = train_data['attention_mask'].squeeze(1).to(device)
            input_id = train_data['input_ids'].squeeze(1).to(device)

            optimizer.zero_grad()
            loss, logits = model(input_id, mask, train_label)

            for i in range(logits.shape[0]):

              logits_clean = logits[i][train_label[i] != -100]
              label_clean = train_label[i][train_label[i] != -100]

              predictions = logits_clean.argmax(dim=1)
              acc = (predictions == label_clean).float().mean()
              total_acc_train += acc
              total_loss_train += loss.item()

            loss.backward()
            optimizer.step()

        model.eval()

        total_acc_val = 0
        total_loss_val = 0

        for val_data, val_label in val_dataloader:

            val_label = val_label.to(device)
            mask = val_data['attention_mask'].squeeze(1).to(device)
            input_id = val_data['input_ids'].squeeze(1).to(device)

            loss, logits = model(input_id, mask, val_label)

            for i in range(logits.shape[0]):

              logits_clean = logits[i][val_label[i] != -100]
              label_clean = val_label[i][val_label[i] != -100]

              predictions = logits_clean.argmax(dim=1)
              acc = (predictions == label_clean).float().mean()
              total_acc_val += acc
              total_loss_val += loss.item()

        train_accuracy = total_acc_train / len(df_train['text'])
        train_loss = total_loss_train / len(df_train['text'])
        val_accuracy = total_acc_val / len(df_val['text'])
        val_loss = total_loss_val / len(df_val['text'])

        print(
            f'Epochs: {epoch_num + 1} | Loss: {train_loss: .3f} | Accuracy: {train_accuracy: .3f} | Val_Loss: {val_loss: .3f} | Accuracy: {val_accuracy: .3f}')


label_all_tokens = False
def align_word_ids(texts):
  
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(1)
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(1 if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids


def evaluate_one_text(model, sentence):


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    text = tokenizer(sentence, padding='max_length', max_length = 512, truncation=True, return_tensors="pt")

    mask = text['attention_mask'].to(device)
    input_id = text['input_ids'].to(device)
    label_ids = torch.Tensor(align_word_ids(sentence)).unsqueeze(0).to(device)

    logits = model(input_id, mask, None)
    logits_clean = logits[0][label_ids != -100]

    predictions = logits_clean.argmax(dim=1).tolist()
    prediction_label = [ids_to_labels[i] for i in predictions]
    # print(sentence)
    # print(prediction_label)
    return(prediction_label)



### CODE ###
testdata = parse_csv_test(testfile)
traindata = parse_csv("train.csv")
validdata = parse_csv("validation.csv")

traindict = {}
validdict = {}

tokens = ""
tokenseq = []
ner_tags = []
ner_tagseq = []
for i in traindata:
  for j in i:
    tokens+= (j[0]) + " "
    ner_tags.append(j[1])
  tokens = tokens[:-1]
  tokenseq.append(tokens)
  ner_tagseq.append(ner_tags)
  tokens = ""
  ner_tags = []

traindict["text"] = tokenseq
traindict["labels"] = ner_tagseq

tokens = ""
tokenseq = []
ner_tags = []
ner_tagseq = []
for i in validdata:
  for j in i:
    tokens+= (j[0]) + " "
    ner_tags.append(j[1])
  tokens = tokens[:-1]
  tokenseq.append(tokens)
  ner_tagseq.append(ner_tags)
  tokens = ""
  ner_tags = []

validdict["text"] = tokenseq
validdict["labels"] = ner_tagseq

unique_labels = set()

for lb in traindict["labels"]:
  [unique_labels.add(i) for i in lb if i not in unique_labels]

labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}

tokenizer = BertTokenizerFast.from_pretrained('Twitter/twhin-bert-base')

# TRAIN

LEARNING_RATE = 5e-3
EPOCHS = 1
# EPOCHS = 10
BATCH_SIZE = 2

model = BertModel()
train_loop(model, traindict, validdict)


# PREDICT

ans = []
for i,j in enumerate(testdata):
  predict = evaluate_one_text(model, j)
  tokens = j.split()
  predLength = len(predict)
  tokenLength = len(tokens)
  if predLength > tokenLength:
    ans.extend(predict[0:tokenLength])
  elif tokenLength > predLength:
    diff = tokenLength - predLength
    ans.extend(predict)
    for i in range(diff):
      ans.append('O')
  else:
    ans.extend(predict)

with open(outputfile, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'label'])  # Write header
    for i in range(len(ans)):
        writer.writerow([i, int(labels_to_ids[ans[i]])])