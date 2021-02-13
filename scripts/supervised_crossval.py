"""A file to run cross-validation on supervised learning methods for predicting MeSH labels from abstracts.
For more information on the code, see the notebook pipeline-to-topics-comparison.ipynb

"""
import argparse

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression

from keras.wrappers.scikit_learn import KerasClassifier

from nltk.corpus import stopwords

from helperfunctions import process_text
import keras_model as km

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import time

from gensim.sklearn_api import Text2BowTransformer
from gensim.matutils import corpus2dense


parser = argparse.ArgumentParser()
parser.add_argument(
    '-o', '--output',
    type=str,
    help="""
    Output filename to save dataframe to as csv. Defaults to 
    'topic-modeled-abstracts-{layer}.csv', where layer corresponds
    to the layer argument value.
    """
)
parser.add_argument(
    '-l', '--layer',
    type=int,
    help="""
    Layer of the MeSH tree that articles came from. Used to create default filenames.
    Defaults to 1.
    """,
    default=1
)
parser.add_argument(
    '-i', '--input',
    type=str,
    help="""
    Input filename to load topic-modeled abstracts from. 
    Defaults to 'topic-modeled-abstracts-{layer}.csv', 
    where layer corresponds to the layer argument value.
    """
)
parser.add_argument(
    '-v', '--verbosity',
    type=int,
    help="""
    Verbosity of printed outputs while training. Either 0 (no outputs) or 1 
    (print outputs). Defaults to 1.
    """,
    default=0
)
parser.add_argument(
    '--keras_epochs',
    type=int,
    help="""
    Number of training epochs for the Keras neural networks. Defaults to 10.
    """,
    default=0
)


args = parser.parse_args()

RANDOM_STATE = 0
NUM_CROSSVAL = 3
N_JOBS_CROSSVAL = 1
VERBOSITY = args.verbosity

TEST_FRACTION = 0.2
FILTER_THRESHOLD = 0.5

NUM_EPOCHS_KERAS = args.keras_epochs
NLP_MODEL = 'en_core_sci_lg'

LAYER = args.layer
if args.input is None:
    INPUT = f'topic-modeled-abstracts-{LAYER}.csv'
else:
    INPUT = args.input

SCORES_FILENAME = f'scores-{LAYER}.csv'
CROSSVALIDATION_PLOT_FILENAME = f'pipeline-plot-{LAYER}.png'

# SHOW_PLOTS = args.show
SHOW_PLOTS = True
# ---------------------

class Text2BowTransformerExtension(Text2BowTransformer):
    '''Effectively this is a Gensim Dictionary, wrapped up to work 
    in a scikit-learn Pipeline with a custom text preprocessor. 
    Allows a custom tokenizer or text-preprocessing 
    function rather than the sklearn defaults.
    '''
    def transform(self, *args, **kwargs):
        result = super().transform(*args, **kwargs)
        return corpus2dense(result, len(self.gensim_model)).T

# ---------------------

df = pd.read_csv(INPUT, index_col=0)
# filter away abstracts with low topic probability
df = df.loc[df.topic_probability >= FILTER_THRESHOLD]
df.head()


abstracts_train, abstracts_test, y_train, y_test = train_test_split(
    df.clean_text[~df.abstract.isna()],
    df.mesh_term_index[~df.abstract.isna()],
    test_size=TEST_FRACTION,
    stratify=df.mesh_term_index[~df.abstract.isna()],
    shuffle=True,
    random_state=RANDOM_STATE
)


stoplist = stopwords.words('english')

vectorizer_params = dict(
    stop_words=stoplist,
    ngram_range=(1, 2),
    max_features=10000 
)


km.prepare_embedding_layer(
    documents=df.clean_text[~df.abstract.isna()],
    nlp_model=NLP_MODEL,
)
output_dim = df.mesh_term_index.nunique()

keras_transformer = km.TokenizeAndPadTransformer()
keras_predictor = KerasClassifier(
    build_fn=km.create_model,
    output_dim=output_dim,
    epochs=NUM_EPOCHS_KERAS,
    batch_size=64,
    verbose=0
)
keras_predictor_lstm = KerasClassifier(
    build_fn=km.create_model_lstm,
    output_dim=output_dim,
    epochs=NUM_EPOCHS_KERAS,
    batch_size=64,
    verbose=0
)


preprocessors = dict(
    keras_transformer=keras_transformer,
    gensim=Text2BowTransformerExtension(tokenizer=process_text),
    count=CountVectorizer(**vectorizer_params),
    tfidf=TfidfVectorizer(**vectorizer_params),
)
predictors = dict(
    keras_predictor=keras_predictor,
    keras_predictor_lstm=keras_predictor_lstm,
    naive_bayes=MultinomialNB(alpha= 0.5, fit_prior=False),
    svm=SVC(C=1, kernel='linear',probability=True, class_weight='balanced',random_state=0), 
    logistic_regression=LogisticRegression(C= 1, max_iter = 100, penalty = 'l2',class_weight='balanced'),
    forest=RandomForestClassifier(max_depth=100, max_features=1000, min_samples_leaf=10,
                       n_estimators=300,class_weight='balanced',random_state=0)
)

cv_scores = {}
scores_array = np.ones(shape=(len(preprocessors), len(predictors))) * np.nan
if VERBOSITY == 1:
    print('Evaluating pipelines:')
for i, prep in enumerate(preprocessors.items()):
    for j, predictor in enumerate(predictors.items()):
        mean_score = np.nan
        if prep[0] == 'keras_transformer' and not predictor[0].startswith('keras'):
            continue
        elif predictor[0].startswith('keras') and not prep[0] == 'keras_transformer':
            continue
        
        n_jobs = N_JOBS_CROSSVAL
        if predictor[0].startswith('keras'):
            # keras won't let me share embedding layer across CPUs
            # so only one model can be trained at a time
            n_jobs = 1

        pipename = '->'.join([prep[0], predictor[0]])
        
        pipeline = Pipeline(steps=[prep, predictor])
        if VERBOSITY == 1:
            print(f'[{pipename}]:\t', end=' ')
        
        start_time = time()
        scores = cross_val_score(
            estimator=pipeline, X=abstracts_train, y=y_train,
            cv=NUM_CROSSVAL,
            n_jobs=n_jobs
            )
        end_time = time()
        
        mean_score = np.mean(scores)
        scores_array[i, j] = mean_score
        cv_scores[pipename] = scores
        if VERBOSITY == 1:
            print(f'runtime: {end_time-start_time:.2f}s\tmean score: {scores.mean()}')


plt.figure(figsize=(15, 6))
plt.boxplot(cv_scores.values(), labels=cv_scores.keys())
plt.xticks(rotation=80)
plt.grid(ls=':')
plt.title('Pipeline Performances')
plt.ylabel('accuracy')

if CROSSVALIDATION_PLOT_FILENAME is not None:
    plt.savefig(CROSSVALIDATION_PLOT_FILENAME, bbox_inches = "tight")

if SHOW_PLOTS:
    plt.show()


scores_df = pd.DataFrame(
    scores_array,
    index=preprocessors.keys(),
    columns=predictors.keys())

scores_df.to_csv(SCORES_FILENAME)