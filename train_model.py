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

#fn to calculate flat accuracy
def flat_accuracy(preds, labels):
    flat_preds = np.argmax(preds, axis=2).flatten()
    flat_labels = labels.flatten()
    return np.sum(flat_preds == flat_labels)/len(flat_labels)

#fn to train
def train(epoch, model, training_loader, device, optimizer):
    model.train()
    f1_scores = []
    for i,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device)
        mask = data['mask'].to(device)
        targets = data['tags'].to(device)

        loss = model(ids, mask, labels = targets)[0]

        # optimizer.zero_grad()
        if i%300==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()

def get_scores(model, testing_loader, device):
    model.eval()
    eval_loss = 0; eval_accuracy = 0
    n_correct = 0; n_wrong = 0; total = 0
    pred_prob_list = []
    predictions , true_labels = [], []
    new_test_sentences = []
    nb_eval_steps, nb_eval_examples = 0, 0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            targets = data['tags'].to(device)

            output = model(ids, mask, labels=targets)
            loss, logits = output[:2]
            logits = logits.detach().cpu().numpy()
            label_ids = targets.to('cpu').numpy()
            pred_prob = [list(pp) for pp in softmax(logits, axis=-1)]
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.append(label_ids)
            accuracy = flat_accuracy(logits, label_ids)
            eval_loss += loss.mean().item()
            eval_accuracy += accuracy
            nb_eval_examples += ids.size(0)
            nb_eval_steps += 1        
        eval_loss = eval_loss/nb_eval_steps
        validation_accuracy = eval_accuracy/nb_eval_steps
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
#         print(f"predictions probability: {pre_prob[0]}")
        pred_tags = [p_i for p in predictions for p_i in p]
        valid_tags = [l_ii for l in true_labels for l_i in l for l_ii in l_i]
        f1 = f1_score(valid_tags, pred_tags, average='macro')
        return [f1, eval_loss, validation_accuracy]

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
                for inner_index, x in enumerate(array_list):
                      try:
                          if (inner_index < no_of_words_array[outer_index] ):
                              if ((np.argmax(x).item()) == 0):
                                  counter_for_inner_array += 1
                                  selected_tokens_arr.append([ids[outer_index][inner_index].item()])
                              elif ((np.argmax(x).item()) == 1 and np.argmax(array_list[inner_index -1 ]).item() == 0) or ((np.argmax(x).item()) == 1 and np.argmax(array_list[inner_index -1 ]).item() == 1 and np.argmax(array_list[inner_index -2 ]).item() == 0):
                                  counter_for_inner_array += 1
                                  selected_tokens_arr.append([ids[outer_index][inner_index - 1].item(), ids[outer_index][inner_index].item()])
                      except:
                          continue
        
        
        print(f"final len of selected_tokens_arry : {len(selected_tokens_arr)}")
        # print(len())
        return selected_tokens_arr

def wrapper_for_train(epochs, model, training_loader, device, optimizer):
    # to train
    avg_time_for_epochs = 0
    total_time = 0
    for epoch in range(epochs):
        start = time.time()
        print(f"start of epoch {epoch} at {datetime.now().time()}")
        train(epoch, model, training_loader, device, optimizer)
        end = time.time()
        print(f"end of epoch {epoch} at {datetime.now().time()}")
        total_time = total_time + (end-start)
        avg_time_for_epochs = (total_time)/(epoch + 1) 
        print(f"average time for epochs: {avg_time_for_epochs}")

    print(f"total time taken was: {total_time}")

def start():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_NAME = 'dmis-lab/biobert-v1.1'
    # MODEL_NAME = 'm3rg-iitd/matscibert'
    model = transformers.BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=3).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    stopwords = nltk.corpus.stopwords.words('english')
    wordnet_lemmatizer = WordNetLemmatizer()

    train_generated = sen_generator("BC2GM/train.tsv", stopwords, wordnet_lemmatizer)
    train_sentences = train_generated[0]
    train_targets = train_generated[1]

    test_generated = sen_generator("BC2GM/test.tsv", stopwords, wordnet_lemmatizer)
    test_sentences = test_generated[0]
    test_targets = test_generated[1]

    training_set = CustomDataset(
        tokenizer=tokenizer,
        sentences=train_sentences,
        labels=train_targets, 
        max_len=200
    )

    testing_set = CustomDataset(
        tokenizer=tokenizer,
        sentences=test_sentences,
        labels=test_targets, 
        max_len=200
    )

    #setting train and testing parameters
    train_params = {'batch_size': 32,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': 16,
                    'shuffle': False,
                    'num_workers': 0
                    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader =  DataLoader(testing_set, **test_params)

    #setting up the optimizer and the learning rate
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=5e-5)

    #train the model for x epochs
    for i in range(10):
        wrapper_for_train(10, model, training_loader, device, optimizer)
        scores = get_scores(model, testing_loader, device)
        file1 = open("scores_file.txt", 'a')
        file1.write(f"After {(i+1) * 10} epochs \n")
        file1.write(f"F1 score : {scores[0]}\n Validation Loss: {scores[1]} \n Validation Accuracy: {scores[2]} \n")
        file1.close()
    #to save the model
    torch.save(model.state_dict(), "100EpochsBioBio")



if __name__=="__main__":
    start()
