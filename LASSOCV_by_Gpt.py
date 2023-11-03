import pandas as pd
import MeCab
from collections import Counter
from gensim import corpora
from gensim.models import LdaModel, TfidfModel
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.feature_extraction.text import CountVectorizer

# Load your dataset
df = pd.read_csv('sk_data.csv')

mecab = MeCab.Tagger()

# Tokenization and Part-of-Speech Tagging
def tokenize(text):
    tokens = []
    pos_tag = []
    morphs = mecab.parse(text)
    lines = morphs.split("\n")
    for line in lines:
        if "\t" in line:
            token, info = line.split("\t")
            token_info = info.split(",")
            surface = token
            pos = token_info[0]
            tokens.append(surface)
            pos_tag.append((surface, pos))
    return tokens, pos_tag

# Define your stop words and filters
stop_pos = ['NR', 'JKC', 'JKS', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'EP', 'EF', 'EC', 'ETN', 'ETM', 'SF', 'SE', 'SS', 'SP', 'SO', 'SW', 'SN', 'MM', 'NA', 'MAG', 'MAJ']
N_pos = ['NNG', 'NNP']
stop_word = ['your', 'stop', 'words', 'here']

# Tokenize and preprocess
def preprocess(text):
    tokens, pos_tag = tokenize(text)
    filtered_text = [(word, pos) for word, pos in pos_tag if len(word) >= 2 and word not in stop_word and pos in N_pos]
    return filtered_text

# Create tokens
df['tokens'] = df['2'].apply(lambda x: [word for word, _ in preprocess(x)])

# Create a dictionary
id2word = corpora.Dictionary(df['tokens'])

# Create a corpus
corpus = [id2word.doc2bow(tokens) for tokens in df['tokens']]

# TF-IDF transformation
tfidf = TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

# Train an LDA model
n_topics = 10  # Number of topics
lda = LdaModel(corpus_tfidf, id2word=id2word, num_topics=n_topics, random_state=100)

# Extract topic-term matrix
topic_term_matrix = lda.get_topics()

# Transform topic-term matrix to a document-term matrix
dtm = np.array([lda[c] for c in corpus_tfidf])

# Perform LassoCV for topic selection
lasso = LassoCV(cv=5, alphas=np.logspace(-6, 6, 13))
lasso.fit(dtm.T)
 vcc                                                                                                                                                                                                    
# Get selected topics
selected_topics = np.where(lasso.coef_ != 0)[0]

# Get coefficients for selected topics
topic_coefficients = lasso.coef_[selected_topics]

# Save the selected topics and coefficients to a DataFrame
selected_topics_df = pd.DataFrame({'Topic Index': selected_topics, 'Coefficient': topic_coefficients})

# Print the selected topics and their coefficients
print(selected_topics_df)

# Save the DataFrame to a CSV file
selected_topics_df.to_csv('selected_topics.csv', index=False)
