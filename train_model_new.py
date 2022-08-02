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

def read_csv_frequency(filename, key):
    data = pd.read_csv(filename, header=0)
    return data[key].to_list()

def plain_sentence_gen(filename, stopwords, wordnet_lemmatizer):
    f = open(filename, "r")
    sentences = []
    sentence_array = []
    sen = []
    for line in f.readlines():
        words = line.split(" ")
        for each in words:
            if each == "\n" and len(sen) != 0:
                sentences.append(' '.join(sen))
                sentence_array.append(sen)
                sen = []
            else:
                if no_punctutation(each) and no_stopwords(each, stopwords):
                    sen.append(wordnet_lemmatizer.lemmatize(each.lower()))
    return [sentences, sentence_array]

def generate_tags(sentences, top_words, bi_grams, tri_grams):
    #print(tri_grams.item())
    print(type(tri_grams))
    #print(type(tri_grams)
    tags = []
    for each in sentences:
        tags.append([2]* len(each))

    for i, each in enumerate(sentences):
        for j, words in enumerate(each):
            a = ' '.join(each[j:j+3])
            if a in tri_grams:
                tags[i][j] = 0
                tags[i][j+1] = 1
                tags[i][j+2] = 1
                
    for i, each in enumerate(sentences):
        for j, words in enumerate(each):
            a = ' '.join(each[j:j+2])
            if (a in bi_grams and tags[i][j] == 2 and tags[i][j+1] == 2):
                tags[i][j] = 0
                tags[i][j+1] = 1

    for i, each in enumerate(sentences):
        for j, words in enumerate(each):
            if ((words in top_words) and tags[i][j] == 2):
                tags[i][j] = 0 
    
    return tags

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

def get_scores(model, testing_loader, device, EPOCHS):
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
        scores = {'f1_score': f1, 'validation_accuracy': validation_accuracy, 'validation_loss': eval_loss, 'epochs': EPOCHS}
        return scores


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
        optimizer.step()

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
        avg_time_for_epochs = (total_time)/(epoch + 1) 
        print(f"average time for epochs: {avg_time_for_epochs}")

    print(f"total time taken was: {total_time}")
    return model

def start(MAX_EPOCHS, EPOCHS, MODEL_OUTPUT, LEARNING_RATE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f1_scores_array = []
    MODEL_NAME = 'bert-base-cased'
    # MODEL_NAME = 'dmis-lab/biobert-v1.1'
    # MODEL_NAME = 'm3rg-iitd/matscibert'
    model = transformers.BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=3).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    stopwords = nltk.corpus.stopwords.words('english')
    wordnet_lemmatizer = WordNetLemmatizer()

    train_generated = plain_sentence_gen("dataset_pure/TT_PDE_Cross-abstracts.txt", stopwords, wordnet_lemmatizer)
    train_sentences = train_generated[0]
    
    top_words = read_csv_frequency("dataset_pure/Tensors_PDE_top_words.csv", "Keyword")
    bi_gram = read_csv_frequency("dataset_pure/Tensors_PDE_bigrams.csv", "Bi-gram")
    tri_gram = read_csv_frequency("dataset_pure/Tensors_PDE_trigrams.csv", "Tri-gram")

    tags = generate_tags(train_generated[1], top_words, bi_gram, tri_gram)

    validation_percent = 0.9
    validation_size = int(validation_percent * len(train_sentences))

    train_sentences = train_sentences[:validation_size]
    train_targets = tags[:validation_size]
    validation_sentences = train_sentences[validation_size:]
    validation_targets = tags[validation_size:]
    
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

    training_set = CustomDataset(
        tokenizer=tokenizer,
        sentences=train_sentences,
        labels=train_targets, 
        max_len=200
    )

    #setting train and testing parameters
    train_params = {'batch_size': 32,
                    'shuffle': True,
                    'num_workers': 0
                    }

    training_loader = DataLoader(training_set, **train_params)

    #setting up the optimizer and the learning rate
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

    #train the model for x epochs
    validation_old_dict = {}
    loops_run = 0
    loop_counter = 0
    while(True):
        temp_dict = {}
        dict_name = 'loop' + str(loop_counter)

        model = wrapper_for_train(EPOCHS, model, training_loader, device, optimizer)
        validation_old_dict[dict_name] = get_scores(model, validation_loader, device, EPOCHS)
        f1_scores_array.append(validation_old_dict[dict_name]['f1_score'])

        loop_counter += 1
        if len(f1_scores_array) >= 3:
            if ((f1_scores_array[-2] - f1_scores_array[-1]) > 0 and (f1_scores_array[-3] - f1_scores_array[-2]) > 0) or (loop_counter*EPOCHS >= MAX_EPOCHS):
                loops_run = loop_counter
                break

    validation_saved = str((loops_run) * EPOCHS) + "_" + MODEL_NAME + "_validation.txt"    
    #to save the model
    try:
        torch.save(model.state_dict(), MODEL_OUTPUT)
    except:
        print("Could not save the model")
    
    file1 = open(validation_saved, "w")
    file1.write(f"{validation_old_dict}")
    file1.write(f"\n\n Max_Epochs: {MAX_EPOCHS} \n Epochs: {EPOCHS} \n Model: {MODEL_NAME}\n")
    file1.write(f"Loops Ran: {loops_run * EPOCHS}\n")
    file1.write(f"Learning Rate: {LEARNING_RATE}\n")
    file1.write(f"Time finished: {datetime.now()}")
    file1.close()



if __name__=="__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--maxEpochs', type=int, default=1, help='Max no of epochs to run')
    parser.add_argument('-e', '--epochs', type=int, default=1, help='No of Epochs')
    parser.add_argument('-s', '--model_output', type=str, default='./ModelOutput', help='Outputfile for the trained Model')
    parser.add_argument('-r', '--learning_rate', type=float, default=5e-5, help='Learning Rate')
    args = parser.parse_args()
    start(args.maxEpochs, args.epochs, args.model_output, args.learning_rate)

