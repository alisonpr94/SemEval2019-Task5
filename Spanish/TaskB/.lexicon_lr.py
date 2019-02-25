# -*- coding: utf-8 -*-

#@author: alison

import re
import string
import pickle
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score

# Etapa de pré-processamento

def clean_tweets(tweet):
    tweet = re.sub('@(\\w{1,15})\b', '', tweet)
    tweet = tweet.replace("via ", "")
    tweet = tweet.replace("RT ", "")
    tweet = tweet.lower()
    return tweet
    
def clean_url(tweet):
    tweet = re.sub('http\\S+', '', tweet, flags=re.MULTILINE)   
    return tweet
    
def remove_stop_words(tweet):
    stops = set(stopwords.words("spanish"))
    stops.update(['.',',','"',"'",'?',':',';','(',')','[',']','{','}'])
    toks = [tok for tok in tweet if not tok in stops and len(tok) >= 3]
    return toks
    
def stemming_tweets(tweet):
    stemmer = SnowballStemmer('spanish')
    stemmed_words = [stemmer.stem(word) for word in tweet]
    return stemmed_words

def remove_number(tweet):
    newTweet = re.sub('\\d+', '', tweet)
    return newTweet

def remove_hashtags(tweet):
    result = ''

    for word in tweet.split():
        if word.startswith('#') or word.startswith('@'):
            result += word[1:]
            result += ' '
        else:
            result += word
            result += ' '

    return result

def preprocessing(tweet, swords = True, url = True, stemming = True, ctweets = True, number = True, hashtag = True):

    if ctweets:
        tweet = clean_tweets(tweet)

    if url:
        tweet = clean_url(tweet)

    if hashtag:
        tweet = remove_hashtags(tweet)
    
    twtk = TweetTokenizer(strip_handles=True, reduce_len=True)

    if number:
        tweet = remove_number(tweet)
    
    tokens = [w.lower() for w in twtk.tokenize(tweet) if w != "" and w is not None]

    if swords:
        tokens = remove_stop_words(tokens)

    if stemming:
        tokens = stemming_tweets(tokens)

    text = " ".join(tokens)

    return text

def bag_of_words(train, test):
    num_features = train.shape[0] + test.shape[0]
    vec = CountVectorizer(min_df=1, max_features=num_features)
    train = vec.fit_transform(train).toarray()
    test = vec.transform(test).toarray()
    return train, test

def save_files(y_test, y_pred, ID, TR, AG):
    #y_pred.dtype = np.int
    #np.savetxt('input/ref/en.tsv', y_test, fmt='%d')

    with open("input/res/es_b.tsv", "w") as file:
        for i in range(len(y_pred)):
            file.write(str(ID[i]))
            file.write('\t')
            file.write(str(y_pred[i]))
            file.write('\t')
            file.write(str(TR[i]))
            file.write('\t')
            file.write(str(AG[i]))
            file.write('\n')

def create_matrix(tweets, pos, neg):
    mat = []

    for tweet in tweets:
        vec = [0 for i in range(3)]
        contPos = 0
        contNeg = 0

        for word in tweet.split():
            if word in pos:
                contPos += 1
            if word in neg:
                contNeg += 1

        if contPos > contNeg:
            vec[0] = 1
        elif contNeg > contPos:
            vec[1] = 1
        else:
            vec[2] = 1
        mat.append(vec)

    return mat


def opinion_lexicon(train, test):
    pos = pd.read_csv("OpinionLexicon/positive-words.csv")
    neg = pd.read_csv("OpinionLexicon/negative-words.csv")

    train = create_matrix(train, list(pos['lexicon_pos']), list(neg['lexicon_neg']))
    test  = create_matrix(test, list(pos['lexicon_pos']), list(neg['lexicon_neg']))

    return train, test

def classify(x_train, y_train, x_test, y_test, flag):
    # Fase de classificação de sentimentos

    clf = LogisticRegression(C=1.0)   # Instância do classificador

    clf.fit(x_train, y_train)   # Fase de treinamento

    # Criando arquivo para salvar modelo treinado
    filename = 'Models/modelLogisticRegressionLex.sav'
    pickle.dump(clf, open(filename, 'wb'))

    y_pred  = clf.predict(x_test)    # Fase de predição, testando dados novos

    if flag:
        # Salvando as classes preditas pelo modelo do algoritmo
        y_pred.dtype = np.int
        np.savetxt('ClassPred/Lex_LR_BoW.txt', y_pred, fmt='%d')

    return y_pred

def main():

    train = pd.read_csv('Dataset/train_es.tsv', delimiter='\t',encoding='utf-8')    # Tem 9000 tweets
    dev = pd.read_csv('Dataset/dev_es.tsv', delimiter='\t',encoding='utf-8')        # Tem 1000 tweets
    trial = pd.read_csv('Dataset/trial_es.tsv', delimiter='\t',encoding='utf-8')    # Tem 100 tweets

    ###########################################################################################################
    
    # Pré-processamento dos tweets

    train_text  = train['text'].map(lambda x: preprocessing(x, swords = True, url = True,
                                    stemming = True, ctweets = True, number = True, hashtag = True))
    y_train = train['HS']
    id_train = train['id']
    tr_train = train['TR']
    ag_train = train['AG']

    test_text  = dev['text'].map(lambda x: preprocessing(x, swords = True, url = True,
                                    stemming = True, ctweets = True, number = True, hashtag = True))
    y_test = dev['HS']
    id_test = dev['id']
    tr_test = dev['TR']
    ag_test = dev['AG']

    ###########################################################################################################
    
    # Bag-of-Words

    # Construindo a BoW para os tweets de treinamento e teste
    x_train, x_test = bag_of_words(train_text, test_text)
    op_train, op_test = opinion_lexicon(train_text, test_text)

    x_train = np.concatenate((x_train, op_train), axis=1)
    x_test  = np.concatenate((x_test, op_test), axis=1)

    ###########################################################################################################

    y_pred  = classify(x_train, y_train, x_test, y_test, True)
    tr_test = classify(x_train, tr_train, x_test, tr_test, False)
    ag_test = classify(x_train, ag_train, x_test, ag_test, False)

    # Salvando arquivos para a avalição em evaluation.py
    save_files(y_test, y_pred, id_test, tr_test, ag_test)

    print("Treinamento finalizado! Testando modelo...")

    #print("F1.........: %f" %(f1_score(y_test, y_pred, average="macro")))
    #print("Precision..: %f" %(precision_score(y_test, y_pred, average="macro")))
    #print("Recall.....: %f" %(recall_score(y_test, y_pred, average="macro")))
    #print("Accuracy...: %f" %(accuracy_score(y_test, y_pred)))

    
if __name__ == '__main__':
    main()