# adapted from https://towardsdatascience.com/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert-41ff868d1794

## for data
import pandas as pd
import numpy as np

## for plotting
import matplotlib.pyplot as plt
import seaborn as sns

## for analysis
import re

## for machine learning
from sklearn import preprocessing, model_selection, feature_extraction, feature_selection, metrics, manifold, naive_bayes, pipeline

## for deep learning
from tensorflow.keras import models, layers, preprocessing as kprocessing
from tensorflow.keras import backend as K

## for W2V
import gensim
import gensim.downloader as gensim_api






def utils_preprocess_ngrams(corpus, ngrams=1, grams_join=" ", lst_ngrams_detectors=[]):
    ## create list of n-grams
    lst_corpus = []
    for string in corpus:
        lst_words = string.split()
        lst_grams = [grams_join.join(lst_words[i:i + ngrams]) for i in range(0, len(lst_words), ngrams)]
        lst_corpus.append(lst_grams)
    
    ## detect common bi-grams and tri-grams
    if len(lst_ngrams_detectors) != 0:
        for detector in lst_ngrams_detectors:
            lst_corpus = list(detector[lst_corpus])
    return lst_corpus



def create_ngrams_detectors(corpus, grams_join=" ", lst_common_terms=[], min_count=5, top=10, figsize=(10,7)):
    ## fit models
    lst_corpus = utils_preprocess_ngrams(corpus, ngrams=1, grams_join=grams_join)
    bigrams_detector = gensim.models.phrases.Phrases(lst_corpus, delimiter=grams_join.encode(), common_terms=lst_common_terms, min_count=min_count, threshold=min_count*2)
    bigrams_detector = gensim.models.phrases.Phraser(bigrams_detector)
    trigrams_detector = gensim.models.phrases.Phrases(bigrams_detector[lst_corpus], delimiter=grams_join.encode(), common_terms=lst_common_terms, min_count=min_count, threshold=min_count*2)
    trigrams_detector = gensim.models.phrases.Phraser(trigrams_detector)

    ## plot
    dtf_ngrams = pd.DataFrame([{"word":grams_join.join([gram.decode() for gram in k]), "freq":v} for k,v in trigrams_detector.phrasegrams.items()])
    dtf_ngrams["ngrams"] = dtf_ngrams["word"].apply(lambda x: x.count(grams_join)+1)
    dtf_ngrams = dtf_ngrams.sort_values(["ngrams","freq"], ascending=[True,False])
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x="freq", y="word", hue="ngrams", dodge=False, ax=ax,
                data=dtf_ngrams.groupby('ngrams')["ngrams","freq","word"].head(top))
    ax.set(xlabel=None, ylabel=None, title="Most frequent words")
    ax.grid(axis="x")
    plt.show()
    return [bigrams_detector, trigrams_detector], dtf_ngrams





def text2seq(corpus, ngrams=1, grams_join=" ", lst_ngrams_detectors=[], top=None, oov=None, maxlen=None):    

    ## detect common n-grams in corpus
    lst_corpus = utils_preprocess_ngrams(corpus, ngrams=ngrams, grams_join=grams_join,  lst_ngrams_detectors=lst_ngrams_detectors)
    print("--- tokenization ---")
    tokenizer = kprocessing.text.Tokenizer(num_words=top,lower=True, split=' ')
    tokenizer.fit_on_texts(lst_corpus)
    dic_vocabulary = tokenizer.word_index
    print(len(dic_vocabulary), "words")
    ## create sequence
    lst_text2seq= tokenizer.texts_to_sequences(lst_corpus)
    ## padding sequence
    print("--- padding to sequence ---")
    X_train = kprocessing.sequence.pad_sequences(lst_text2seq, 
                        maxlen=maxlen, padding="post", truncating="post")
    print(X_train.shape[0], "sequences of length", X_train.shape[1]) 
    ## plot heatmap
    fig, ax = plt.subplots(figsize=(15,5))
    sns.heatmap(X_train==0, vmin=0, vmax=1, cbar=False, ax=ax).set_title('Sequences Overview')
    plt.show()
    return {"X":X_train, "tokenizer":tokenizer, "dic_vocabulary":dic_vocabulary}




'''
Plot loss and metrics of keras training.
'''
def utils_plot_keras_training(training):
    metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15,3))
    
    ## training
    ax[0].set(title="Training")
    ax11 = ax[0].twinx()
    ax[0].plot(training.history['loss'], color='black')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax11.plot(training.history[metric], label=metric)
    ax11.set_ylabel("Score", color='steelblue')
    ax11.legend()
    
    ## validation
    ax[1].set(title="Validation")
    ax22 = ax[1].twinx()
    ax[1].plot(training.history['val_loss'], color='black')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax22.plot(training.history['val_'+metric], label=metric)
    ax22.set_ylabel("Score", color="steelblue")
    plt.show()



def fit_bow(corpus, vectorizer=None, vocabulary=None):
    ## vectorizer
    vectorizer = feature_extraction.text.TfidfVectorizer(max_features=None, ngram_range=(1,1), vocabulary=vocabulary) if vectorizer is None else vectorizer
    vectorizer.fit(corpus)
    
    ## sparse matrix
    print("--- creating sparse matrix ---")
    X = vectorizer.transform(corpus)
    print("shape:", X.shape)
    
    ## vocabulary
    print("--- creating vocabulary ---") if vocabulary is None else print("--- used vocabulary ---")
    dic_vocabulary = vectorizer.vocabulary_   #{word:idx for idx, word in enumerate(vectorizer.get_feature_names())}
    print(len(dic_vocabulary), "words")
    
    ## text2tokens
    print("--- tokenization ---")
    tokenizer = vectorizer.build_tokenizer()
    preprocessor = vectorizer.build_preprocessor()
    lst_text2tokens = []
    for text in corpus:
        lst_tokens = [dic_vocabulary[word] for word in tokenizer(preprocessor(text)) if word in dic_vocabulary]
        lst_text2tokens.append(lst_tokens)
    print(len(lst_text2tokens), "texts")
    
    ## plot heatmap
    fig, ax = plt.subplots(figsize=(15,5))
    sns.heatmap(X.todense()[:,np.random.randint(0,X.shape[1],100)]==0, vmin=0, vmax=1, cbar=False, ax=ax).set_title('Sparse Matrix Sample')
    plt.show()
    return {"X":X, "lst_text2tokens":lst_text2tokens, "vectorizer":vectorizer, "dic_vocabulary":dic_vocabulary, "X_names":vectorizer.get_feature_names()}







