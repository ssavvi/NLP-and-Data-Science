""" A file with helper functions for the project pipeline, 
so that the notebook etc. are easier to read.
"""
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import word_tokenize
import numpy as np

# from gensim.utils import simple_preprocess
# `simple_preprocess` splits up hyphenated words, lowercases them, 
# chooses a maximum and minimum word length. While convenient, it's
# probably better to consider each step independently so that one 
# can determine what's important and what isn't.

lemmatizer = WordNetLemmatizer()
stoplist = stopwords.words('english')

def process_text(
    text,
    hyphenated_words='split',
    remove_numerals=True,
    min_length=2,
    max_length=None,
    remove_punctuation=True,
    return_string=False,
    delimiter=' '
):
    """
    params:  
    `hyphenated_words` = str in {'split' (default), 'remove', 'keep'} or None, in
    which case the default ('split') is used.
    `remove_numerals` = bool, whether to remove numeral tokens.
    `min_length`, `max_length` = int or None. If int is given, denotes the min/max 
    length of allowed tokens (inclusive).  
    `remove_punctuation` = bool, whether to remove punctuation from *within* tokens 
    and remove tokens which were entirely punctuation.  
    `return_string` = bool, whether to return a list of tokens or a single string. 
    `delimiter` = str, delimiter to use when rejoining tokens to return a single 
    string.
    """
    # Get words from text
    tokens = word_tokenize(text)
    
    # Handle hyphenated words
    if hyphenated_words is None:
        # use the default value
        hyphenated_words = 'split'
    if hyphenated_words == 'split':
        # words are split at the hyphens into individual tokens
        tokens = sum(
            [token.split('-') for token in tokens], []
        )
    elif hyphenated_words == 'remove':
        # hyphenated words are removed
        tokens = [token for token in tokens if not '-' in token]
    elif hyphenated_words != 'keep':
        # tokens are kept as is
        raise Exception(f'{hyphenated_words} is an invalid value for `hyphenated_words` parameter.')
        
    # Handle numbers
    if remove_numerals:
        tokens = [token for token in tokens if not token.isnumeric()]
    
    # Handle word lengths
    if max_length is None:
        max_length = np.inf
    if min_length is None:
        min_length = 0
    tokens = [token for token in tokens
              if len(token) >= min_length
              and len(token) <= max_length
             ]
    
    # Handle punctuation
    # Hyphens aren't removed
    if remove_punctuation:
        no_punc_tokens = []
        for token in tokens:
            new_token = token
            for punc in punctuation:
                # remove punctuation from token
                if punc == '-': 
                    # Don't remove hyphens here
                    continue
                new_token = new_token.replace(punc, '')
            if new_token:
                # keep token if it hasn't become empty
                no_punc_tokens.append(new_token)
        tokens = no_punc_tokens
    
    # Lemmatize and lowercase tokens
    # catch verbs:
    tokens = [
        lemmatizer.lemmatize(t.lower(), 'v')
        for t in tokens
    ]
    # catch nouns:
    tokens = [
        lemmatizer.lemmatize(t.lower(), 'n')
        for t in tokens
    ]
    
    # Strip whitespace
    tokens = [token.strip() for token in tokens]
    
    # Remove stopwords
    tokens = [token for token in tokens if not token in stoplist]

    tokens=' '.join(map(str, tokens))
   
    return tokens if not return_string else delimiter.join(tokens)
    


# kept for the sake of comparison
from gensim import utils
import gensim.parsing.preprocessing as gsp

filters = [
           gsp.strip_tags, 
           gsp.strip_punctuation,
           gsp.strip_multiple_whitespaces,
           gsp.strip_numeric,
           gsp.remove_stopwords, 
           gsp.strip_short, 
           gsp.stem_text
          ]

def process_text_gensim(s):
    #s = s.lower()
    s = utils.to_unicode(s)
    for f in filters:
        s = f(s)
    return s
