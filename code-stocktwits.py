
# Import Common modules
from tqdm.notebook import tqdm
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

sns.set(style='white', context='notebook', palette='deep')

# Set Random Seed

rand_seed = 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)



# Define metrics
from sklearn.metrics import accuracy_score, f1_score
import scikitplot as skplt

# Here, use F1 Macro to evaluate the model.
def metric(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, f1

scoring = {'Accuracy': 'accuracy', 'F1': 'f1_macro'}
refit = 'F1'

import json

def load_tweet(filename):
    ''' 
        Input:
            - filename
        Output:
            - a dataframe for the loaded data
    '''
    with open(filename, 'r') as f:
        twits = json.load(f)

    for i, message in enumerate(twits):
  
      try:
          if (message['entities']['sentiment']['basic'] == 'Bullish'):
              twits[i]['entities']['sentiment']  = 4
          elif (message['entities']['sentiment']['basic'] == 'Bearish'):
              twits[i]['entities']['sentiment'] = 2
          else:
              twits[i]['entities']['sentiment'] = 0
      except:
              twits[i]['entities']['sentiment'] = 0

    # logger.debug(twits[:10])
    # logger.info("The number of twits is: {}".format(len(twits)))
      messages = [twit['body'] for twit in twits]
    # #  scale the sentiments to 0 to 4 for use in the network
      sentiments = [twit['entities']['sentiment'] for twit in twits]
    
    return messages,sentiments



# Load data
filename = 'TSLA.json'
messages,sentiments= load_tweet(filename)



import re

def preprocess(message):
    """
    This function takes a string as input, then performs these operations: 
        - lowercase
        - remove URLs
        - remove ticker symbols 
        - removes punctuation
        - tokenize by splitting the string on whitespace 
        - removes any single character tokens
    
    Parameters
    ----------
        message : The text message to be preprocessed.
        
    Returns
    -------
        tokens: The preprocessed text into tokens.
    """ 
    # Lowercase the Stocktwit message
    text = message.lower()
    
    # Remove URLs and add space 
    text = re.sub('https?:\/\/[a-zA-Z0-9@:%._\/+~#=?&;-]*', ' ', text)
    
    # remove ticker symbols and add space
    text = re.sub('\$[a-zA-Z0-9]*', ' ', text)
    
    # remove StockTwits usernames and add space. The usernames are any word that starts with @.
    text = re.sub('\@[a-zA-Z0-9]*', ' ', text)

    # remove everything not a letter or apostrophe with a space
    text = re.sub('[^a-zA-Z\']', ' ', text)

    # Remove single letter words
    text = ' '.join( [w for w in text.split() if len(w)>1] )
    
    return text

test_message = "RT @google Our annual looked at the year in a Google's blogging (and beyond) http://t.co/sptHOAh8 $GOOG"
print(preprocess(test_message))

# Process for all messages
preprocessed = [preprocess(message) for message in tqdm(messages)]

from collections import Counter
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def tokenize_text(text, option):
    '''
    Tokenize the input text as per specified option
      Use python split() function
       Use NLTK word_tokenize()
       Use NLTK word_tokenize(),
       remove stop words and apply lemmatization
    '''
    if option == 1:
        return text.split()
    elif option == 2:
        return re.findall(r'\b([a-zA-Z]+n\'t|[a-zA-Z]+\'s|[a-zA-Z]+)\b', text)
    elif option == 3:
        return [word for word in word_tokenize(text) if (word.isalpha()==1)]
    elif option == 4:
        words = [word for word in word_tokenize(text) if (word.isalpha()==1)]
        # Remove stop words
        stop = set(stopwords.words('english'))
        words = [word for word in words if (word not in stop)]
        # Lemmatize words (first noun, then verb)
        wnl = nltk.stem.WordNetLemmatizer()
        lemmatized = [wnl.lemmatize(wnl.lemmatize(word, 'n'), 'v') for word in words]
        return lemmatized
    else:
        logger.warn("Please specify option value between 1 and 4")

tokenize_text(preprocessed[0], 4)

# Create vocab
def create_vocab(messages, show_graph=False):
    corpus = []
    for message in tqdm(messages, desc="Tokenizaing"):
        tokens = tokenize_text(message, 3) # Use option 3
        corpus.extend(tokens)
    logger.info("The number of all words: {}".format(len(corpus)))

    # Create Counter
    counts = Counter(corpus)
    logger.info("The number of unique words: {}".format(len(counts)))

    # Create BoW
    bow = sorted(counts, key=counts.get, reverse=True)
    logger.info("Top 40 frequent words: {}".format(bow[:40]))

    # Indexing vocabrary, starting from 1.
    vocab = {word: ii for ii, word in enumerate(counts, 1)}
    id2vocab = {v: k for k, v in vocab.items()}

    if show_graph:
        from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
        # Generate Word Cloud image
        text = " ".join(corpus)
        stopwords = set(STOPWORDS)
        stopwords.update(["will", "report", "reporting", "market", "stock", "share"])

        wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=100, background_color="white", collocations=False).generate(text)
        plt.figure(figsize=(15,7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

        # Show most frequent words in a bar graph
        most = counts.most_common()[:80]
        x, y = [], []
        for word, count in most:
            if word not in stopwords:
                x.append(word)
                y.append(count)
        plt.figure(figsize=(12,10))
        sns.barplot(x=y, y=x)
        plt.show()

    return vocab

vocab= create_vocab(preprocessed, True)

## Create token id list
#token_ids = [[vocab[word] for word in text_words] for text_words in tokenized] # comment out to save memory

tmp_dict = {'org message': messages, 'sentence': preprocessed, 'label': sentiments}
tmp_df = pd.DataFrame(tmp_dict).sample(n=20, random_state=rand_seed)

# Samples
pd.set_option('display.max_colwidth', 200)
tmp_df

# Change the table display config back
pd.set_option('display.max_colwidth', 50)

# Create a dataframe for training data
word_cnt = [len(tokenize_text(x, 3)) for x in tqdm(preprocessed)]

# Use tweets having 5 or more words. Do not resample for balancing data here.
train_dict = {'text': preprocessed, 'label': sentiments, 'count': word_cnt}
train_df = pd.DataFrame(train_dict)
train_df = train_df.loc[train_df['count'] >= 5]
train_df.reset_index(drop=True, inplace=True)
logger.info("The total number of input data: {}".format(len(train_df)))

# Display the distribution graph
fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(17,5))
#sns.countplot(x='label', data=train_df, ax=ax1)
ax1.set_title('The number of data for each label', fontsize=14)
sns.histplot([len(x) for x in train_df['text']], ax=ax2, bins=100)
ax2.set_title('The number of letters in each data', fontsize=14)
ax2.set_xlim(0,150)
ax2.set_xlabel('number of letters')
ax2.set_ylabel("")
sns.histplot(train_df['count'], ax=ax3, bins=100)
ax3.set_title('The number of words in each data', fontsize=14)
ax3.set_xlim(0,40)
ax3.set_xlabel('number of words')
ax3.set_ylabel("")

plt.show()

# Import Pytorch modules
# import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, TensorDataset, Dataset

from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from transformers import AdamW as AdamW_HF, get_linear_schedule_with_warmup

# Define a DataSet Class which simply return (x, y) pair
class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.datalist=[(x[i], y[i]) for i in range(len(y))]
    def __len__(self):
        return len(self.datalist)
    def __getitem__(self,idx):
        return self.datalist[idx]

# Data Loader
def create_data_loader(X, y, indices, batch_size, shuffle):
    X_sampled = np.array(X, dtype=object)[indices]
    y_sampled = np.array(y)[indices].astype(int)
    dataset = SimpleDataset(X_sampled, y_sampled)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

from sklearn.model_selection import StratifiedShuffleSplit

def train_cycles(X_all, y_all, vocab, num_samples, model_type, epochs, patience, batch_size, seq_len, lr, clip, log_level):
    result = pd.DataFrame(columns=['Accuracy', 'F1(macro)', 'Total_Time', 'ms/text'], index=num_samples)

    for n in num_samples:
        
        # Stratified sampling
        train_size = n / len(y_all)
        sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size, test_size=train_size*0.2 , random_state=rand_seed)
        train_indices, valid_indices = next(sss.split(X_all, y_all))

        # Sample input data
        train_loader = create_data_loader(X_all, y_all, train_indices, batch_size, True)
        valid_loader = create_data_loader(X_all, y_all, valid_indices, batch_size, False)

        if model_type == 'LSTM':
            model = TextClassifier(len(vocab)+1, embed_size=512, lstm_size=1289, dense_size=0, output_size=5, lstm_layers=4, dropout=0.2)
            model.embedding.weight.data.uniform_(-1, 1)
        elif model_type == 'BERT':
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

        start_time = time.perf_counter() # use time.process_time() for CPU time
        acc, f1, model_trained = train_nn_model(model, model_type, train_loader, valid_loader, vocab, epochs, patience, batch_size, seq_len, lr, clip, log_level)
        end_time = time.perf_counter() # use time.process_time() for CPU time
        duration = end_time - start_time
        logger.info("Process Time (sec): {}".format(duration))
        result.loc[n] = (round(acc,4), round(f1,4), duration, duration/n*1000)

    return result, model_trained

def train_nn_model(model, model_type, train_loader, valid_loader, vocab, epochs, patience, batch_size, seq_len, lr, clip, log_level):
    # Set variables
    logger = set_logger('sa_tweet_inperf', log_level)
    num_total_opt_steps = int(len(train_loader) * epochs)
    eval_every = len(train_loader) // 5
    warm_up_proportion = 0.1
    logger.info('Total Training Steps: {} ({} batches x {} epochs)'.format(num_total_opt_steps, len(train_loader), epochs))

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = AdamW_HF(model.parameters(), lr=lr, correct_bias=False) 
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_total_opt_steps*warm_up_proportion, num_training_steps=num_total_opt_steps)  # PyTorch scheduler
    criterion = nn.NLLLoss()

    # Set Train Mode
    model.train()

    # Initialise
    acc_train, f1_train, loss_train, acc_valid, f1_valid, loss_valid = [], [], [], [], [], []
    best_f1, early_stop, steps = 0, 0, 0
    class_names = ['0:Very Negative','1:Negative', '2:Neutral', '3:Positive', '4:Very Positive']

    for epoch in tqdm(range(epochs), desc="Epoch"):
       
        # Initialise
        loss_tmp, loss_cnt = 0, 0
        y_pred_tmp, y_truth_tmp = [], []
        hidden = model.init_hidden(batch_size) if model_type == "LSTM" else None

        for i, batch in enumerate(train_loader):
            text_batch, labels = batch
            # Skip the last batch of which size is not equal to batch_size
            if labels.size(0) != batch_size:
                break
            steps += 1
           
            # Reset gradient
            model.zero_grad()
            optimizer.zero_grad()

            # Initialise after the previous training
            if steps % eval_every == 1:
                y_pred_tmp, y_truth_tmp = [], []

            if model_type == "LSTM":
                # Tokenize the input and move to device
                text_batch = tokenizer_lstm(text_batch, vocab, seq_len, padding='left').transpose(1,0).to(device)
                labels = torch.tensor(labels, dtype=torch.int64).to(device)

                # Creating new variables for the hidden state to avoid backprop entire training history
                hidden = tuple([each.data for each in hidden])
                for each in hidden:
                    each.to(device)

                # Get output and hidden state from the model, calculate the loss
                logits, hidden = model(text_batch, hidden)
                loss = criterion(logits, labels)
                
            elif model_type == 'BERT':
                # Tokenize the input and move to device
                # Tokenizer Parameter
                param_tk = {
                    'return_tensors': "pt",
                    'padding': 'max_length',
                    'max_length': seq_len,
                    'add_special_tokens': True,
                    'truncation': True
                }
                text_batch = tokenizer_bert(text_batch, **param_tk).to(device)
                labels = torch.tensor(labels, dtype=torch.int64).to(device)

                # Feedforward prediction
                loss, logits = model(**text_batch, labels=labels)

            y_pred_tmp.extend(np.argmax(F.softmax(logits, dim=1).cpu().detach().numpy(), axis=1))
            y_truth_tmp.extend(labels.cpu().numpy())

            # Back prop
            loss.backward()

            # Training Loss
            loss_tmp += loss.item()
            loss_cnt += 1

            # Clip the gradient to prevent the exploading gradient problem in RNN/LSTM
            nn.utils.clip_grad_norm_(model.parameters(), clip)

            # Update Weights and Learning Rate
            optimizer.step()
            scheduler.step()


            #################### Evaluation ####################
            if (steps % eval_every == 0) or ((steps % eval_every != 0) and (steps == len(train_loader))):
                # Evaluate Training
                acc, f1 = metric(y_truth_tmp, y_pred_tmp)
                acc_train.append(acc)
                f1_train.append(f1)
                loss_train.append(loss_tmp/loss_cnt)
                loss_tmp, loss_cnt = 0, 0

                # y_pred_tmp = np.zeros((len(y_valid), 5))
                y_truth_tmp, y_pred_tmp = [], []

                # Move to Evaluation Mode
                model.eval()

                with torch.no_grad():
                    for i, batch in enumerate(valid_loader):
                        text_batch, labels = batch
                        # Skip the last batch of which size is not equal to batch_size
                        if labels.size(0) != batch_size:
                            break

                        if model_type == "LSTM":
                            # Tokenize the input and move to device
                            text_batch = tokenizer_lstm(text_batch, vocab, seq_len, padding='left').transpose(1,0).to(device)
                            labels = torch.tensor(labels, dtype=torch.int64).to(device)

                            # Creating new variables for the hidden state to avoid backprop entire training history
                            hidden = tuple([each.data for each in hidden])
                            for each in hidden:
                                each.to(device)

                            # Get output and hidden state from the model, calculate the loss
                            logits, hidden = model(text_batch, hidden)
                            loss = criterion(logits, labels)
                
                        elif model_type == 'BERT':
                            # Tokenize the input and move to device
                            text_batch = tokenizer_bert(text_batch, **param_tk).to(device)
                            labels = torch.tensor(labels, dtype=torch.int64).to(device)
                            # Feedforward prediction
                            loss, logits = model(**text_batch, labels=labels)
                    
                        loss_tmp += loss.item()
                        loss_cnt += 1

                        y_pred_tmp.extend(np.argmax(F.softmax(logits, dim=1).cpu().detach().numpy(), axis=1))
                        y_truth_tmp.extend(labels.cpu().numpy())
                        # logger.debug('validation batch: {}, val_loss: {}'.format(i, loss.item() / len(valid_loader)))

                acc, f1 = metric(y_truth_tmp, y_pred_tmp)
                logger.debug("Epoch: {}/{}, Step: {}, Loss: {:.4f}, Acc: {:.4f}, F1: {:.4f}".format(epoch+1, epochs, steps, loss_tmp, acc, f1))
                acc_valid.append(acc)
                f1_valid.append(f1)
                loss_valid.append(loss_tmp/loss_cnt)
                loss_tmp, loss_cnt = 0, 0

                # Back to train mode
                model.train()

        #################### End of each epoch ####################

        # Show the last evaluation metrics
        logger.info('Epoch: %d, Loss: %.4f, Acc: %.4f, F1: %.4f, LR: %.2e' % (epoch+1, loss_valid[-1], acc_valid[-1], f1_valid[-1], scheduler.get_last_lr()[0]))

        # Plot Confusion Matrix
        y_truth_class = [class_names[int(idx)] for idx in y_truth_tmp]
        y_predicted_class = [class_names[int(idx)] for idx in y_pred_tmp]
        
        titles_options = [("Actual Count", None), ("Normalised", 'true')]
        for title, normalize in titles_options:
            disp = skplt.metrics.plot_confusion_matrix(y_truth_class, y_predicted_class, normalize=normalize, title=title, x_tick_rotation=75)
        plt.show()

        # plot training performance
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))
        ax1.set_title("Losses")
        ax1.set_xlabel("Validation Cycle")
        ax1.set_ylabel("Loss")
        ax1.plot(loss_train, 'b-o', label='Train Loss')
        ax1.plot(loss_valid, 'r-o', label='Valid Loss')
        ax1.legend(loc="upper right")
        
        ax2.set_title("Evaluation")
        ax2.set_xlabel("Validation Cycle")
        ax2.set_ylabel("Score")
        ax2.set_ylim(0,1)
        ax2.plot(acc_train, 'y-o', label='Accuracy (train)')
        ax2.plot(f1_train, 'y--', label='F1 Score (train)')
        ax2.plot(acc_valid, 'g-o', label='Accuracy (valid)')
        ax2.plot(f1_valid, 'g--', label='F1 Score (valid)')
        ax2.legend(loc="upper left")

        plt.show()

        # If improving, save the number. If not, count up for early stopping
        if best_f1 < f1_valid[-1]:
            early_stop = 0
            best_f1 = f1_valid[-1]
        else:
            early_stop += 1

        # Early stop if it reaches patience number
        if early_stop >= patience:
            break

        # Prepare for the next epoch
        if device == 'cuda:0':
            torch.cuda.empty_cache()
        model.train()

    return acc, f1, model

# Define LSTM Model
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, lstm_size, dense_size, output_size, lstm_layers=2, dropout=0.1):
       
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.lstm_size = lstm_size
        self.dense_size = dense_size
        self.output_size = output_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, lstm_size, lstm_layers, dropout=dropout, batch_first=False)
        self.dropout = nn.Dropout(dropout)
        if dense_size == 0:
            self.fc = nn.Linear(lstm_size, output_size)
        else:
            self.fc1 = nn.Linear(lstm_size, dense_size)
            self.fc2 = nn.Linear(dense_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    def init_hidden(self, batch_size):
        
        #Initialize the hidden state
        
        weight = next(self.parameters()).data
        hidden = (weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_(),
                  weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_())
                
        return hidden

    def forward(self, nn_input_text, hidden_state):
        """
        Perform a forward pass of the model on nn_input
        """
        batch_size = nn_input_text.size(0)
        nn_input_text = nn_input_text.long()
        embeds = self.embedding(nn_input_text)
        lstm_out, hidden_state = self.lstm(embeds, hidden_state)
        # Stack up LSTM outputs, apply dropout
        lstm_out = lstm_out[-1,:,:]
        lstm_out = self.dropout(lstm_out)
        # Dense layer
        if self.dense_size == 0:
            out = self.fc(lstm_out)
        else:
            dense_out = self.fc1(lstm_out)
            out = self.fc2(dense_out)
        # Softmax
        logps = self.softmax(out)

        return logps, hidden_state

# Define a tokenizer
def tokenizer_lstm(X, vocab, seq_len, padding):
    X_tmp = np.zeros((len(X), seq_len), dtype=np.int64)
    for i, text in enumerate(X):
        tokens = tokenize_text(text, 3) 
        token_ids = [vocab[word] for word in tokens]
        end_idx = min(len(token_ids), seq_len)
        if padding == 'right':
            X_tmp[i,:end_idx] = token_ids[:end_idx]
        elif padding == 'left':
            start_idx = max(seq_len - len(token_ids), 0)
            X_tmp[i,start_idx:] = token_ids[:end_idx]

    return torch.tensor(X_tmp, dtype=torch.int64)

# Define the training parameters
num_samples = [500,1000,5000,8000]
epochs=5
patience=3
batch_size=64
seq_len = 30
lr=3e-4
clip=5
log_level=logging.DEBUG

# Run!
result_lstm, model_trained_lstm = train_cycles(train_df['text'], train_df['label'], vocab, num_samples, 'LSTM', epochs, patience, batch_size, seq_len, lr, clip, log_level)
result_lstm

# Use pretrained model
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

# Tokenizer Parameter
param_tk = {
    'return_tensors': "pt",
    'padding': 'max_length',
    'max_length': seq_len,
    'add_special_tokens': True,
    'truncation': True
}

# Test the model and tokenizer
inputs = tokenizer_bert("Hello, my dog is cute", **param_tk)

print('inputs: \n', inputs)
print('\ndecoded: \n',tokenizer_bert.decode(inputs['input_ids'].squeeze(0)))

labels = torch.tensor([1]).unsqueeze(0)
print('\nlabels: ', labels)

outputs = bert_model(**inputs, labels=labels)
print('\noutputs: length=', len(outputs))
print(outputs)

loss = outputs[0]
logits = outputs[1]

print('loss: ', loss.detach())
print('logits: ', logits.detach())

print(len(outputs))
print('outputs: \n',outputs)
print('outputs(detached): \n', outputs[0].detach())

# Define the training parameters
num_samples = [500,1000, 5000,8000,12000]
epochs=5
patience=3
batch_size=64
seq_len = 30
lr=2e-5
clip=1.0
log_level=logging.DEBUG

# Run!
result_bert, model_trained_bert = train_cycles(train_df['text'], train_df['label'], vocab, num_samples, 'BERT', epochs, patience, batch_size, seq_len, lr, clip, log_level)
result_bert

# Save the model and show the result
torch.save(model_trained_lstm.state_dict(),  'stocktwit_bert.dict')
result_bert

# If disconnected after the training complete, just recreate the result_df rather than running the training again...
disconnected = False
if disconnected:
    n_trains = [500,1000, 5000,8000,12000]
    result_lstm = pd.DataFrame(columns=['Accuracy', 'F1(macro)', 'Total_Time'], index=n_trains)
    result_lstm.loc[1000] = (0.4115,	0.1166,	4.02131)
    result_lstm.loc[5000] = (0.4969,	0.4542,	13.189)
    result_lstm.loc[10000] = (0.5514,	0.4988,	23.8845)
    result_lstm.loc[100000] = (0.6856,	0.6548,	273.114)
    result_lstm.loc[500000] = (0.7593,	0.7375,	2704.45)
    result_lstm['ms/data'] = result_lstm['Total_Time'] / result_lstm.index * 1000
result_lstm

# If disconnected after the training complete, just recreate the result_df rather than running the training again...
disconnected = False
if disconnected:
    n_trains = [1000, 5000]
    result_bert = pd.DataFrame(columns=['Accuracy', 'F1(macro)', 'Total_Time'], index=n_trains)
    result_bert.loc[1000] = (0.465,	0.2876,	25.8849)
    result_bert.loc[5000] = (0.5355,	0.4475,	56.1339)
    result_bert.loc[10000] = (0.6482,	0.6039,	110.48)
    result_bert.loc[100000] = (0.7453,	0.7223,	1092.68)
    result_bert.loc[500000] = (0.7889,	0.7693,	5443.28)
    result_bert['ms/data'] = result_bert['Total_Time'] / result_bert.index * 1000
result_bert

# Plot Result Graph
def plot_result(column):
    result_accuracy = pd.DataFrame(data=[result_lstm[column], result_bert[column]], index=['LSTM', 'BERT']).T
    ax = result_accuracy.plot.bar(rot=0, figsize=(10,7))
    ax.set_title(column, fontsize=16)
    ax.set_xlabel('Training Data Count')
    for p in ax.patches:
        ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.005, p.get_height() * 1.005))

plot_result('Accuracy')
plot_result('F1(macro)')
plot_result('ms/text')

with open('twits.json', 'r') as f:
    test_data = json.load(f)

def twit_stream():
    for twit in test_data:
        yield twit

next(twit_stream())

import torch.nn.functional as F

def predict(text, model, tokenizer):
    """ 
    Make a prediction on a single sentence.

    Parameters
    ----------
        text : The string to make a prediction on.
        model : The model to use for making the prediction.
        vocab : Dictionary for word to word ids. The key is the word and the value is the word id.

    Returns
    -------
        pred : Prediction vector
    """   
    text = preprocess(text)
    inputs = tokenizer(text, 
                   return_tensors="pt", 
                   padding='max_length',
                   max_length=96,
                   add_special_tokens=True,
                   truncation=True)

    outputs = model(**inputs)[0].detach()    
    pred = F.softmax(outputs, dim=1)
    
    return pred



from transformers import BertTokenizer, BertForSequenceClassification, BertModel
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
# model.load_state_dict(torch.load('stocktwit_bert.dict'))
model.eval()

# # Check
text = "Google is working on self driving cars, I'm bullish on $goog"
predict(text, model, tokenizer)

def score_twits(stream, model, tokenizer, universe):
    """ 
    Given a stream of twits and a universe of tickers, return sentiment scores for tickers in the universe.
    """
    class_names = ['0:Very Negative', '2:Neutral', '4:Very Positive']
    for twit in stream:

        # Get the message text twits[i]['entities']['sentiment']
        text = twit['body']
        if len(tokenizer.tokenize(preprocess(text))) < 10:
            continue
        symbols = re.findall('\$[A-Z]{2,4}', text)
        score = predict(text, model, tokenizer)
        score = np.round(score.tolist(), 4).squeeze()
        prediction = class_names[np.argmax(score)] + " {:.1f}%".format(np.max(score)*100)

        for symbol in symbols:
            if symbol in universe:
                yield {'symbol': symbol, 'pred': prediction, 'score': score, 'text': text}

# Select Universe
universe = {'$BBRY', '$AAPL', '$AMZN', '$BABA', '$YHOO', '$LQMT', '$FB', '$GOOG', '$BBBY', '$JNUG', '$SBUX', '$MU'}
score_stream = score_twits(twit_stream(), model, tokenizer, universe)

# Process
for i in range(8):
    print(next(score_stream))
    i+=1
