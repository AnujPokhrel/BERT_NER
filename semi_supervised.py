from ossaudiodev import SNDCTL_COPR_SENDMSG
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
import sys

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
    print("new epoch")
    for i,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device)
        mask = data['mask'].to(device)
        targets = data['tags'].to(device)

        loss = model(ids, mask, labels = targets)[0]

        # optimizer.zero_grad()
        if i%300==0:
            print(f'Epoch: {epoch} Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
    
    return model

def wrapper_for_train(epochs, model, training_loader, device, optimizer):
    # to train
    avg_time_for_epochs = 0
    total_time = 0
    for epoch in range(epochs):
        start = time.time()
        print(f"start of epoch {epoch} at {datetime.now().time()}")
        model = train(epoch, model, training_loader, device, optimizer)
        end = time.time()
        print(f"end of epoch {epoch} at {datetime.now().time()}")
        total_time = total_time + (end-start)
        avg_time_for_epochs = total_time/(epoch + 1)
        print(f"avg time for epochs: {avg_time_for_epochs}")
    
    print(f"total time taken : {total_time}")
    return model

def get_scores(model, testing_loader, device):
    model.eval()
    eval_loss = 0; eval_accuracy = 0
    predictions , true_labels = [], []
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
        pred_tags = [p_i for p in predictions for p_i in p]
        valid_tags = [l_ii for l in true_labels for l_i in l for l_ii in l_i]
        f1 = f1_score(valid_tags, pred_tags, average='macro')
        scores = {'f1_score': f1, 'validation_accuracy': eval_loss, 'validation_loss': validation_accuracy}
        return scores


#get f1 score, print accuracy and loss
def get_new_dataset(model, testing_loader, device, test_targets, prob_threshold):
    model.eval()
    eval_loss = 0; eval_accuracy = 0
    predictions , true_labels = [], []
    new_test_sentences, new_test_targets, for_train_sentences, for_train_targets = [], [], [], []
    nb_eval_steps, nb_eval_examples = 0, 0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            targets = data['tags'].to(device)
            test_sentences = data['sentences']

            output = model(ids, mask, labels=targets)
            loss, logits = output[:2]
            logits = logits.detach().cpu().numpy()
            label_ids = targets.to('cpu').numpy()

            no_of_words_array = []
            no_of_words = 0
            for x in enumerate(mask.cpu().numpy()):
                for each in x[1]:
                    if each == 1:
                        no_of_words+= 1
                    else:
                        no_of_words_array.append(no_of_words)
                        no_of_words = 0
                        break
            
            pred_prob = [list(pp) for pp in softmax(logits, axis=-1)]
            
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.append(label_ids)
            accuracy = flat_accuracy(logits, label_ids)
            eval_loss += loss.mean().item()
            eval_accuracy += accuracy
            nb_eval_examples += ids.size(0)
            nb_eval_steps += 1

            for outer_index, array_list in enumerate(pred_prob):
                max_val = []
                average_max_val, no_of_words_for_index = 0, 0
                test_target_temp = []
                for inner_index, x in enumerate(array_list):
                    try:
                        if (inner_index <= no_of_words_array[outer_index]):
                            max_val.append(np.max(array_list[1]))
                            test_target_temp.append(test_targets[outer_index][inner_index])
                        else:
                            break
                    except:
                        break
                
                if len(max_val) != 0:
                    average_max_val = sum(max_val)/len(max_val)
                
                if average_max_val > prob_threshold:
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
        f1 = f1_score(valid_tags, pred_tags, average='macro')
        scores = {'f1_score': f1, 'validation_accuracy': eval_loss, 'validation_loss': validation_accuracy}

        return [for_train_sentences, for_train_targets, new_test_sentences, new_test_targets, scores]

def start(LOOPS, EPOCHS, SEMI_SUP_OTPT, VALIDATION_OTPT, PROB_THRES ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"The device is sss: {device}")
    MODEL_NAME = 'dmis-lab/biobert-v1.1'
    # MODEL_NAME = 'm3rg-iitd/matscibert'
    #MODEL_NAME = 'bert-base-cased'
    model = transformers.BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=3).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    stopwords = nltk.corpus.stopwords.words('english')
    wordnet_lemmatizer = WordNetLemmatizer()

    train_generated = sen_generator("BC2GM/train.tsv", stopwords, wordnet_lemmatizer)
    test_generated = sen_generator("BC2GM/test.tsv", stopwords, wordnet_lemmatizer)
    train_sentences = train_generated[0]
    train_targets = train_generated[1]
    
    test_sentences = test_generated[0]
    test_targets = test_generated[1]

    validation_percent = 0.9
    validation_size = int(validation_percent * len(train_sentences))
    validation_sentences = train_sentences[validation_size:]
    validation_targets = train_targets[validation_size:]

    train_sentences = train_sentences[0:validation_size]
    train_targets = train_targets[0:validation_size]
    
    print(len(train_sentences))
    print(len(test_sentences))
    #setting up the optimizer and the learning rate
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=1e-5) 

    validation_set = CustomDataset(
            tokenizer=tokenizer,
            sentences=validation_sentences,
            labels=validation_targets, 
            max_len=200
    )
    
    validation_params = {'batch_size': 16,
                    'shuffle': False,
                    'num_workers': 0
                    }
    
    validation_loader =  DataLoader(validation_set, **validation_params)

    result_dict, validation_dict = {}, {}
    result_dict['epoch-1'] = {'scores': {'f1_score': 0, 'validation_loss': 0, 'validation_accuracy': 0}, 'length_of_train': len(train_sentences), 'length_of_test': len(test_sentences)}
    for i in range(LOOPS):
        temp_dict = {}
        dict_name = 'epoch' + str(i)

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
        
        model = wrapper_for_train(EPOCHS, model, training_loader, device, optimizer)

        prob_dataset = get_new_dataset(model, testing_loader, device, test_targets, PROB_THRES)
        train_sentences.extend(prob_dataset[0])
        train_targets.extend(prob_dataset[1])
        test_sentences = prob_dataset[2]
        test_targets = prob_dataset[3]
        
        temp_dict['scores'] = prob_dataset[4]
        temp_dict['length_of_train'] = len(train_sentences)
        temp_dict['length_of_test'] = len(test_sentences)
        result_dict[dict_name] = temp_dict

        validation_dict[dict_name] = get_scores(model, validation_loader, device)
        
    # torch.save(model.state_dict(), "200EpochsBioBioSemiSuper")
    file1 = open(SEMI_SUP_OTPT, 'w')
    file1.write(f"{result_dict}")
    file1.write(f"\n\n Loops: {LOOPS}\n Epochs: {EPOCHS} \n {SEMI_SUP_OTPT}\n {VALIDATION_OTPT}\n")
    file1.write(f"Probability Threshold: {PROB_THRES}\n Model: {MODEL_NAME}")
    file1.close()

    file1 = open(VALIDATION_OTPT, "w")
    file1.write(f"{validation_dict}")
    file1.write(f"\n\n Loops: {LOOPS}\n Epochs: {EPOCHS} \n {SEMI_SUP_OTPT}\n {VALIDATION_OTPT}\n")
    file1.write(f"Probability Threshold: {PROB_THRES}\n Model: {MODEL_NAME}")
    file1.close()


if __name__=="__main__":
    LOOPS = 0
    EPOCHS = 0
    SEMI_SUP_OTPT = 'semisuper_output.txt'
    VALIDATION_OTPT = 'validation_output.txt'
    PROB_THRES = 0.45
    try:
        LOOPS = int(sys.argv[1])
        EPOCHS = int(sys.argv[2])
        SEMI_SUP_OTPT = sys.argv[3]
        VALIDATION_OTPT = sys.argv[4]
        PROB_THRES = float(sys.argv[5])
        print(LOOPS, type(LOOPS))
        print(EPOCHS, type(EPOCHS))
        print(SEMI_SUP_OTPT, type(SEMI_SUP_OTPT))
        print(VALIDATION_OTPT, type(VALIDATION_OTPT))
        print(PROB_THRES, type(PROB_THRES))
    except IndexError:
        print("5 Arguments needed\n")
        print("1) No of Loops\n")
        print("2) No of Epochs\n")
        print("3) Outputfile for Semisupervised data\n")
        print("4) Outputfile for valiation scores data\n")
        print("5) Probability threshold\n")
        sys.exit()
    start(LOOPS, EPOCHS, SEMI_SUP_OTPT, VALIDATION_OTPT, PROB_THRES)
