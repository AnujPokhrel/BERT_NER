import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from sklearn.metrics import f1_score
import pandas as pd
import sklearn
import tensorflow as tf
from scipy.special import softmax 
from datetime import datetime
import time
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support
import string
import nltk
from nltk.stem import WordNetLemmatizer
import argparse


#converting B,I and O to numerical values
def tag_converter(t):
    if t=='B':
        return 0
    elif t=='I':
        return 1
    else:
        return 2

def no_punctutation(word):
  for each in word:
    if each in string.punctuation:
      return 0
  
  return 1

def no_stopwords(text, stopwords):
    if text in stopwords:
      return 0
    else:
      return 1

#get individual sentences from the Data
def sen_generator(filename, stopwords, wordnet_lemmatizer):
  f = open(filename, "r")
  sentences = []
  targets = []
  sen = []
  t = []
  for line in f.readlines():
      word = line.split('\t')[0]
      if word=='\n':
          sentences.append(' '.join(sen))
          targets.append(t)
          sen = []
          t = []
      else:
          if no_punctutation(word) and no_stopwords(word, stopwords):
            target = line.split('\t')[1].strip('\n')
            sen.append(wordnet_lemmatizer.lemmatize(word.lower()))            
            # sen.append(word.lower())
            t.append(tag_converter(target))
  return [sentences, targets]

#class for creating the custom dataset
class CustomDataset(Dataset):
    def __init__(self, tokenizer, sentences, labels, max_len):
        self.len = len(sentences)
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        sentence = str(self.sentences[index])
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        e_sentences = self.sentences[index]
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        label = self.labels[index]
        label.extend([1]*200)
        label=label[:200]

        return {
            'sentences': e_sentences,
            'ids': torch.tensor(ids),
            'mask': torch.tensor(mask),
            'tags': torch.tensor(label)
        } 
    
    def __len__(self):
        return self.len

#get f1 score, print accuracy and loss
def get_ner_tokens(model, testing_loader, device):
    model.eval()
    pred_prob_list = []
    predictions , true_labels = [], []
    new_test_sentences = []
    selected_tokens_arr = []
    counter_for_inner_array = 0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            targets = data['tags'].to(device)
            sentences = data['sentences']
            
            output = model(ids, mask, labels=targets)
            loss, logits = output[:2]
            logits = logits.detach().cpu().numpy()
            label_ids = targets.to('cpu').numpy()

            no_of_words_array = []
            no_of_words = 0
            for x in enumerate(mask.cpu().numpy()):
                # print(x)
                for each in x[1]:
                    if each == 1:
                        no_of_words+= 1
                    else:
                        no_of_words_array.append(no_of_words)
                        no_of_words = 0
                        break

            pred_prob = [list(pp) for pp in softmax(logits, axis=-1)]
            pred_prob_list.extend(pred_prob)
            
            for outer_index, array_list in enumerate(pred_prob):
                # average_max_val, max_val, no_of_words_for_index = 0, 0, 0
                b_is_set = False
                b_set_on = 0
                for inner_index, x in enumerate(array_list):
                    try:
                        if (inner_index < no_of_words_array[outer_index] ):
                            if ((np.argmax(x).item()) == 0):
                                counter_for_inner_array += 1
                                selected_tokens_arr.append([ids[outer_index][inner_index].item()])
                                b_is_set = True
                                b_set_on = inner_index
                            elif ((np.argmax(x).item()) == 1 and b_is_set == True):
                                counter_for_inner_array += 1
                                selected_tokens_arr[-1].append(ids[outer_index][inner_index].item())
                            elif ((np.argmax(x).item()) == 2 and b_is_set == True):
                                b_is_set = False
                    except:
                        continue
        
        
        print(f"final len of selected_tokens_arry : {len(selected_tokens_arr)}")
        # print(len())
        return selected_tokens_arr

def start(trained_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MODEL_NAME = 'dmis-lab/biobert-v1.1'
    # MODEL_NAME = 'm3rg-iitd/matscibert'
    MODEL_NAME = 'bert-base-cased'
    model = transformers.BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=3).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

    model.load_state_dict(torch.load(trained_model))

    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    stopwords = nltk.corpus.stopwords.words('english')
    wordnet_lemmatizer = WordNetLemmatizer()

    test_generated = sen_generator("BC4CHEMD/test.tsv", stopwords, wordnet_lemmatizer)
    test_sentences = test_generated[0]
    test_targets = test_generated[1]

    testing_set = CustomDataset(
        tokenizer=tokenizer,
        sentences=test_sentences,
        labels=test_targets, 
        max_len=200
    )

    test_params = {'batch_size': 16,
                    'shuffle': False,
                    'num_workers': 0
                    }

    # training_loader = DataLoader(training_set, **train_params)
    testing_loader =  DataLoader(testing_set, **test_params)

    ner_tokens = get_ner_tokens(model, testing_loader, device)

    file1 = open("Ner_tokens.txt", 'a')
    for en, each in enumerate(ner_tokens):
        file1.write(f"{tokenizer.decode(each)} \n")
        
    file1.close()


if __name__=="__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', type=str, default='100Epochs', help='Trained Model')
    args = parser.parse_args()
    start(args.model)
