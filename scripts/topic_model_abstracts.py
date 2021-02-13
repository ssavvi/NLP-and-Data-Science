"""A file to handle topic modeling the pubmed abstracts, as well as constructing plots.


"""
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from sklearn.metrics import confusion_matrix, adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, mutual_info_score

from helperfunctions import process_text

from sklearn.manifold import TSNE
from gensim.matutils import sparse2full
from scipy.spatial.distance import pdist, squareform
from matplotlib import cm
from scipy.spatial import ConvexHull

# parse arguments
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
    '-i', '--input',
    type=str,
    help="""
    Input filename to load abstracts from. Defaults to 'abstracts-{layer}.csv', 
    where layer corresponds to the layer argument value.
    """
)
parser.add_argument(
    '-s', '--show',
    action="store_true",
    help="""Use this flag if plots should be shown. 
    This is independent of whether they are saved.
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
    '--tp',
    action="store_true",
    help="""
    This flag determines whether to use a topic prior. Defaults to False.
    """
)
parser.add_argument(
    '--wp',
    action="store_true",
    help="""
    This flag determines whether to use a word prior. Defaults to False.
    """
)
parser.add_argument(
    '-n', '--num_passes',
    type=int,
    help="""
    Number of passes (like epochs) to run the topic modeller for.
    Defaults to 10.
    """,
    default=10
)
parser.add_argument(
    '--num_random',
    type=int,
    help="""
    Number of random reassignments of labels when computing p-values.
    Defaults to 10000.
    """,
    default=10000
)
parser.add_argument(
    '--pval',
    type=str,
    help="""
    Filename for the histogram of scores used for computing p-values.
    Defaults to 'p-value-plot-{layer}.png' where layer corresponds
    to the layer argument value.
    """
)


args = parser.parse_args()

LAYER = args.layer
if args.input is None:
    INPUT = f'abstracts-{LAYER}.csv'
else:
    INPUT = args.input
if args.output is None:
   
    SAVE_FILE_NAME=f'topic-modeled-abstracts-{LAYER}.csv'
else:
    SAVE_FILE_NAME = args.output

SHOW_PLOTS = args.show

NUM_PASSES = args.num_passes
USE_TOPIC_PRIOR = args.tp
USE_WORD_PRIOR = args.wp
NUM_RANDOM_ARRANGEMENTS = args.num_random

if args.pval is None:
    P_VALUE_PLOTS_FILENAME = f'p-value-plot-{LAYER}.png'
else:
    P_VALUE_PLOTS_FILENAME = args.pval


# ---------------------
labels = {}
labels[0] = [
    "Anatomy",
    "Organisms",
    "Diseases",
    "Chemicals and Drugs",
    "Analytical, Diagnostic and Therapeutic Techniques, and Equipment",
    "Psychiatry and Psychology",
    "Phenomena and Processes",
    "Disciplines and Occupations",
    "Anthropology, Education, Sociology, and Social Phenomena",
    "Technology, Industry, and Agriculture",
    "Humanities",
    "Information Science",
    "Named Groups",
    "Health Care",
    "Publication Characteristics",
    "Geographicals"
]
labels[1] = [
    "Physical Phenomena",
    "Chemical Phenomena",
    "Metabolism",
    "Cell Physiological Phenomena",
    "Genetic Phenomena",
    "Microbiological Phenomena",
    "Physiological Phenomena",
    "Reproductive and Urinary Physiological Phenomena",
    "Circulatory and Respiratory Physiological Phenomena",
    "Digestive System and Oral Physiological Phenomena",
    "Musculoskeletal and Neural Physiological Phenomena",
    "Immune System Phenomena",
    "Integumentary System Physiological Phenomena",
    "Ocular Physiological Phenomena",
    "Plant Physiological Phenomena",
    "Biological Phenomena",
    "Mathematical Concepts"
]
labels[2] = [
    "Bacterial Physiological Phenomena", 
    "Biofilms",
    "Catabolite Repression", 
    "Drug Resistance, Microbial", 
    "Germ-Free Life",
    "Hemadsorption",
    "Host Microbial Interactions",
    "Host-Pathogen Interactions",
    "Microbial Interactions",
    "Microbial Viability",
    "Microbiota",
    "Nitrogen Fixation",
    "Toxin-Antitoxin Systems",
    "Virus Physiological Phenomena",
    "Virulence"
]

labels[3] = [
    "Antibody-Dependent Enhancement",
    "Cell Transformation, Viral ",
    "Cytopathogenic Effect, Viral",
    "Drug Resistance, Viral",
    "Hemagglutination, Viral",
    "Inclusion Bodies, Viral",
    "Viral Interference",
    "Viral Load",
    "Viral Tropism",
    "Virus Attachment",
    "Virus Inactivation",
    "Virus Integration",
    "Virus Internalization",
    "Virus Latency ",
    "Virus Release",
    "Virus Replication",
    "Virus Uncoating"
]



df = pd.read_csv(INPUT)
tokenized_corpus = df.abstract[~df.abstract.isna()].apply(process_text)

def Convert(string): 
    li = list(string.split(" ")) 
    return li

tokenized_corpus2 = [Convert(x) for x in tokenized_corpus] 

id2word = Dictionary(tokenized_corpus2)
bow_corpus = [
    id2word.doc2bow(token_list)
    for token_list in tokenized_corpus2
]


alpha = 'symmetric'
if USE_TOPIC_PRIOR:
    alpha = df.mesh_term.value_counts().values.copy()
    alpha = alpha / alpha.sum()

eta = None
if USE_WORD_PRIOR:
    from gensim.matutils import corpus2dense
    eta = corpus2dense(bow_corpus, len(id2word)).sum(axis=0)
    eta = eta / eta.sum()

lda_model = LdaModel(
    corpus=bow_corpus,
    id2word=id2word,
    num_topics=df.mesh_term.nunique(),
    alpha=alpha,
    eta=eta,
    passes=NUM_PASSES
)


def get_dominant_topic(bow, return_prob=True):
    """Returns the topic with highest probability for input `bow`. 
    If `return_prob==True`, also returns the probability for the 
    corresponding topic.
    """
    topics = sorted(
        lda_model.get_document_topics(bow, minimum_probability=0.0),
        key=lambda x: x[1]
        )
    return topics[-1] if return_prob else topics[-1][0]

# we can only reasonably get the topics of non-empty documents
df['bow_length'] = 0.0
df['dominant_topic'] = np.nan
df.loc[~df.abstract.isna(), 'bow_length'] = [len(bow) for bow in bow_corpus]
dominant_topics = [
    get_dominant_topic(bow)
    for bow in bow_corpus
    if len(bow) > 0
    ]
dominant_topics = np.array(dominant_topics)
df.loc[df['bow_length'] > 0, 'dominant_topic'] = dominant_topics[:, 0]
df.loc[df['bow_length'] > 0, 'topic_probability'] = dominant_topics[:, 1]
# it helps to have the mesh terms be indices in the same range as the topics
# labels (mesh terms) are filtered because some labels may have no documents
labels_filtered = [l for l in labels[LAYER] if l in df.mesh_term.value_counts().index]
df['mesh_term_index'] = df.mesh_term.apply(
    labels_filtered.index
)
df['clean_text'] = tokenized_corpus

# Plot the confusion matrix of abstracts with computed topics
# against the MeSH terms
# plt.figure()
conf_mat = confusion_matrix(
    df.mesh_term_index[~df.dominant_topic.isna()],
    df.dominant_topic[~df.dominant_topic.isna()]
    )
plt.imshow(conf_mat)
plt.xticks(range(lda_model.num_topics))
plt.yticks(range(lda_model.num_topics))
plt.title('Confusion Matrix')
plt.colorbar()

if SHOW_PLOTS:
    plt.show()


scores = [
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    mutual_info_score
    ]

mesh_labels = df.mesh_term_index[~df.dominant_topic.isna()]
dominant_topics = df.dominant_topic[~df.dominant_topic.isna()]
num_topics_or_labels = lda_model.num_topics

# Use the same bias in the topic model and mesh terms
# in the random sampling
random_arrangements = np.random.choice(
    a=lda_model.num_topics,  # chooses from np.arange(lda_model.num_topics)
    size=(len(mesh_labels), NUM_RANDOM_ARRANGEMENTS),
    p=alpha
    ) if USE_TOPIC_PRIOR else np.random.choice(
    a=lda_model.num_topics,
    size=(len(mesh_labels), NUM_RANDOM_ARRANGEMENTS)
    )

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, score in enumerate(scores):
    random_scores = []
    true_score = score(mesh_labels, dominant_topics)
    for sample_idx in range(NUM_RANDOM_ARRANGEMENTS):
        random_score = score(random_arrangements[:, sample_idx], dominant_topics)
        random_scores.append(random_score)
    # estimate p-value as probability that a random arrangement has a higher score
    p_val = (np.array(random_scores) > true_score).astype(int).sum() / NUM_RANDOM_ARRANGEMENTS
    axes[i % 2][int(i >= 2)].hist(random_scores, bins=int(np.sqrt(NUM_RANDOM_ARRANGEMENTS)))
    axes[i % 2][int(i >= 2)].set_title(
        score.__name__.replace('_', ' ').upper() \
        + '\n' + f'true value: {true_score:.4f}'\
        + '\n' + f'p-value: {p_val}'
        )

# if p-value == 0.0, then of all the random arrangments, NONE had a value
# higher than that of the true score. So the true p-value is estimated as
# less than 1 / NUM_RANDOM_ARRANGEMENTS == 1/10000

# true value = a measure of correlation between
# mesh label and topic assignments
fig.tight_layout(pad=2.0)

if P_VALUE_PLOTS_FILENAME is not None:
    fig.savefig(P_VALUE_PLOTS_FILENAME)
if SHOW_PLOTS:
    plt.show()
    

df.to_csv(SAVE_FILE_NAME)
