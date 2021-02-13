"""A file to handle the downloading of pubmed abstracts using their MeSH IDs.

"""
import argparse

from Bio import Entrez
from bs4 import BeautifulSoup

import pandas as pd
import numpy as np

from itertools import count

Entrez.email = "suzana,savvi27@gmail.com"

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '-o', '--output',
    type=str,
    help="""
    Output filename to save dataframe to as csv. Defaults to 'abstracts-{layer}.csv', 
    where layer corresponds to the layer argument value.
    """
)
parser.add_argument(
    '-l', '--layer',
    type=int,
    help="Layer of the MeSH tree to download articles from. Defaults to 1",
    default=1
)
parser.add_argument(
    '-m', '--maximum',
    type=int,
    help="Maximum number of articles per MeSH term. Defaults to 1000.",
    default=1000
)
parser.add_argument(
    '--max_query',
    type=int,
    help="""
    Maximum number of articles to query for by MeSH term before filtering. 
    Defaults to 5000
    """,
    default=5000
)
parser.add_argument(
    '--min_abstract_len',
    type=int,
    help="""
    Minimum length of abstracts. Shorter abstracts are dropped. 
    Defaults to 5
    """,
    default=250
)

args = parser.parse_args()

LAYER = args.layer
if args.output is None:
    SAVE_FILE_NAME = f'abstracts-{LAYER}.csv' 
else:
    SAVE_FILE_NAME = args.output

MAX_PER_MESH = args.maximum
MAX_ARTICLES_QUERIED = args.max_query
MIN_ABSTRACT_LENGTH = args.min_abstract_len

# define constants
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



retmax = 1000
# download article IDs
grouped_articles = {}
for i, mesh_term in enumerate(labels[LAYER]):
    all_article_ids = []
    for j in count():
        new_ids_list = Entrez.read(Entrez.esearch(
            db='pubmed', term=mesh_term, field='MESH', retmax=retmax,
            retstart=j*retmax
        ))['IdList']
        all_article_ids += new_ids_list
        if len(new_ids_list) < retmax:  # reached end of pagination
            break
        if len(all_article_ids) >= MAX_ARTICLES_QUERIED:
            break
    grouped_articles[mesh_term] = all_article_ids

# remove duplicates
for i, mesh_term1 in enumerate(labels[LAYER][:-1]):
    for mesh_term2 in labels[LAYER][i+1:]:
        term1_set = set(grouped_articles[mesh_term1])
        overlap = term1_set.intersection(grouped_articles[mesh_term2])
        for article_id in overlap:
            grouped_articles[mesh_term1].remove(article_id)
            grouped_articles[mesh_term2].remove(article_id)

# prune articles to max count
for k in grouped_articles.keys():
    grouped_articles[k] = grouped_articles[k][:MAX_PER_MESH]

# create a dataframe of the articles
grouped_articles_reversed = dict()
for mesh_term, article_id_list in grouped_articles.items():
    grouped_articles_reversed.update(
        {article_id: mesh_term for article_id in article_id_list}
    )

# group the articles into a dataframe
df = pd.DataFrame.from_dict(
    grouped_articles_reversed,
    orient='index',
    columns=['mesh_term']
    )
df.reset_index(inplace=True)
df.rename(columns={"index": "pubmed_id"}, inplace=True)

# download the article abstracts from pubmed
# in fixed batch sizes


all_abstracts = []
batch_size = 1000
for i in range(0, df.shape[0], batch_size):
    print(
    f'Fetching articles {i} to {min(i + batch_size,df.shape[0])} of {df.shape[0]}'
    )
    all_articles_soup = BeautifulSoup(
        Entrez.efetch(
            db='pubmed',
            id=','.join(df.pubmed_id.iloc[i:i+batch_size]),
            retmode='xml'
        )
    )
    articles_split = all_articles_soup.find_all('pubmedarticle')
    articles = []
    for soup in articles_split:
        abstract = soup.find('abstract')
        if abstract:
            texts = [
                x.get_text(separator='')
                for x in abstract.find_all('abstracttext')
            ]
            articles.append('\n'.join(texts))
        else:
            articles.append('')
    all_abstracts += articles

df['abstract'] = [abstract.strip() for abstract in all_abstracts]
# normalise the near-empty abstracts to NaNs
nan_mask = df.abstract.apply(
    lambda x: len(x) < MIN_ABSTRACT_LENGTH
)
df.loc[nan_mask, 'abstract'] = np.nan

df.to_csv(SAVE_FILE_NAME)
