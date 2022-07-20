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

#get f1 score, print accuracy and loss
def get_new_dataset(model, testing_loader):
    model.eval()
    eval_loss = 0; eval_accuracy = 0
    n_correct = 0; n_wrong = 0; total = 0
    pred_prob_list = []
    predictions , true_labels = [], []
    new_test_sentences, new_test_targets, for_train_sentences, for_train_targets = [], [], [], []
    nb_eval_steps, nb_eval_examples = 0, 0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            targets = data['tags'].to(device)
            test_sentences = data['sentences']
            # test_targets = data['o_tags']

            output = model(ids, mask, labels=targets)
            loss, logits = output[:2]
            logits = logits.detach().cpu().numpy()
            label_ids = targets.to('cpu').numpy()

            # print(label_ids)
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
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.append(label_ids)
            accuracy = flat_accuracy(logits, label_ids)
            eval_loss += loss.mean().item()
            eval_accuracy += accuracy
            nb_eval_examples += ids.size(0)
            nb_eval_steps += 1

            for outer_index, array_list in enumerate(pred_prob):
                average_max_val, max_val, no_of_words_for_index = 0, 0, 0
                test_target_temp = []
                for inner_index, x in enumerate(array_list):
                    # print(inner_index)
                    # print(no_of_words_array[outer_index])
                    try:
                        if (inner_index <= no_of_words_array[outer_index]):
                            # print(array_list[1])
                            max_val = max_val + np.max(array_list[1])
                            test_target_temp.append(test_targets[outer_index][inner_index])
                        else:
                            no_of_words_for_index = no_of_words_array[outer_index]
                            break
                    except:
                        break
                
                if no_of_words_for_index != 0:
                    average_max_val = max_val/no_of_words_for_index

                if average_max_val > 0.5:
                    # print("Here we go")
                    # print(f"{test_sentences[outer_index]}")
                    # print(average_max_val)
                    # print(f"{targets[outer_index]}")
                    # print(f"{pred_prob[outer_index]}")
                    for_train_sentences.append(test_sentences[outer_index])
                    for_train_targets.append(test_target_temp)
                else: 
                    new_test_sentences.append(test_sentences[outer_index])
                    new_test_targets.append(test_target_temp)
        
            
        eval_loss = eval_loss/nb_eval_steps
        validation_accuracy = eval_accuracy/nb_eval_steps
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

        pred_tags = [p_i for p in predictions for p_i in p]
        valid_tags = [l_ii for l in true_labels for l_i in l for l_ii in l_i]
        f1 = f1_score(valid_tags, pred_tags, average=None)

        return [for_train_sentences, for_train_targets, new_test_sentences, new_test_targets, [f1, eval_loss, validation_accuracy]]

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

    train_generated = sen_generator("/content/drive/MyDrive/NERdata/BC2GM/train.tsv", stopwords, wordnet_lemmatizer)
    train_generated_1 = sen_generator("/content/drive/MyDrive/NERdata/BC2GM/test.tsv", stopwords, wordnet_lemmatizer)
    train_sentences = train_generated[0]
    train_sentences.extend(train_generated_1[0])
    train_targets = train_generated[1]
    train_targets.extend(train_generated_1[1])

    train_percent = 0.3
    train_size = int(train_percent*len(train_sentences))

    test_sentences = train_sentences[train_size:]
    test_targets = train_targets[train_size:]

    train_sentences = train_sentences[0:train_size]
    train_targets = train_targets[0:train_size]
    
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=5e-5)
    f1_scores, length_of_train, length_of_test = [], [], []

    for i in range(2):
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
        

        #train the model for x epochs
        if i == 0:
            wrapper_for_train(1, model, training_loader, device, optimizer)
        
        wrapper_for_train(1, model, training_loader, device, optimizer)
        prob_dataset = get_new_dataset(model, testing_loader)
        f1_scores.append(prob_dataset[4])
        length_of_train.append(len(prob_dataset[0]))
        length_of_test.append(len(prob_dataset[2]))
        train_sentences.extend(prob_dataset[0])
        train_targets.extend(prob_dataset[1])
        test_sentences = prob_dataset[2]
        test_targets = prob_dataset[3]
        #to save the model
    
    torch.save(model.state_dict(), "50EpochsBioBioSemiSuper")
    file1 = open("semisuperdata.txt", 'a')
    file1.write(f"F1 Scores array: \n{f1_scores}\n\n\n\n")
    file1.write(f"Length of train: \n {length_of_train}\n\n\n")
    file1.write(f"Length of test: \n {length_of_test}\n\n\n")
    file1.close()
    



if __name__=="__main__":
    start()
