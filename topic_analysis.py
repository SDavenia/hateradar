import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary

df = pd.read_csv("data/dataset/posts_and_comments.csv")
print(df.columns)

df_post = df[['post_id', 'subreddit', 'permalink', 'post_created', 'post_author',
       'title', 'selftext', 'url', 'source_file']].drop_duplicates()
print(df_post.shape)

df_post = df_post.drop_duplicates(subset=['post_id'], keep='last')
print(df_post.shape)

titles = df_post["title"].tolist()




nlp = spacy.load("it_core_news_sm")
italian_stopwords = list(nlp.Defaults.stop_words)

# ─────────────────────────────────────────
# 1. VECTORIZE
# ─────────────────────────────────────────
# TF-IDF works better than raw counts for NMF on short texts
vectorizer = TfidfVectorizer(
    stop_words=italian_stopwords,
    min_df=1,
    ngram_range=(1, 2),
    max_features=500       # cap vocabulary for small datasets
)

X = vectorizer.fit_transform(titles)  # titles from your existing df_post pipeline
feature_names = vectorizer.get_feature_names_out()

print(f"Vocabulary size: {len(feature_names)}")
print(f"Document-term matrix shape: {X.shape}")


# ─────────────────────────────────────────
# 2. FIND BEST NUMBER OF TOPICS
# ─────────────────────────────────────────

# Convert titles to tokenized form
tokenized = [t.lower().split() for t in titles]
dictionary = Dictionary(tokenized)

best_k, best_score = None, -1
for k in range(4, 10):
    nmf = NMF(n_components=k, random_state=42)
    W = nmf.fit_transform(X)
    H = nmf.components_

    # Extract top 10 words per topic as lists
    topics_words = [
        [feature_names[i] for i in topic.argsort()[-10:][::-1]]
        for topic in H
    ]
    cm = CoherenceModel(
        topics=topics_words,
        texts=tokenized,
        dictionary=dictionary,
        coherence="c_v"
    )
    score = cm.get_coherence()
    print(f"k={k} | coherence: {score:.4f}")
    if score > best_score:
        best_score, best_k = score, k

print(f"\nBest k: {best_k}")


# ─────────────────────────────────────────
# 3. FIT NMF
# ─────────────────────────────────────────
N_TOPICS = best_k  # tweak this — try 5, 6, 7, 8

nmf_model = NMF(
    n_components=N_TOPICS,
    random_state=42,
    max_iter=500
)

W = nmf_model.fit_transform(X)   # shape: (n_docs, n_topics)  — doc-topic matrix
H = nmf_model.components_        # shape: (n_topics, n_words) — topic-word matrix

# ─────────────────────────────────────────
# 4. INSPECT TOPICS — top words per topic
# ─────────────────────────────────────────
TOP_N = 10

print("\n" + "="*50)
print("NMF TOPICS — Top words")
print("="*50)

for topic_idx, topic_vec in enumerate(H):
    top_indices = topic_vec.argsort()[-TOP_N:][::-1]
    top_words = [feature_names[i] for i in top_indices]
    print(f"\nTopic {topic_idx}: {' | '.join(top_words)}")

# ─────────────────────────────────────────
# 5. ASSIGN TOPICS TO DOCUMENTS
# ─────────────────────────────────────────
# Each doc gets the topic with the highest weight
doc_topics = W.argmax(axis=1)
doc_scores = W.max(axis=1)    # confidence score for the assigned topic

df_post["nmf_topic"] = doc_topics
df_post["nmf_score"] = doc_scores

print("\n" + "="*50)
print("TOPIC DISTRIBUTION")
print("="*50)
print(df_post["nmf_topic"].value_counts().sort_index())

# ─────────────────────────────────────────
# 6. INSPECT DOCS PER TOPIC
# ─────────────────────────────────────────
print("\n" + "="*50)
print("REPRESENTATIVE DOCS PER TOPIC")
print("="*50)

TOP_DOCS = 3  # how many docs to show per topic

for topic_id in range(N_TOPICS):
    topic_docs = df_post[df_post["nmf_topic"] == topic_id]
    top_docs = topic_docs.nlargest(TOP_DOCS, "nmf_score")

    # top words for reference
    top_words = [feature_names[i] for i in H[topic_id].argsort()[-5:][::-1]]

    print(f"\n── Topic {topic_id} [{' | '.join(top_words)}] ──")
    for _, row in top_docs.iterrows():
        print(f"  [{row['nmf_score']:.2f}] {row['title']}")

# ─────────────────────────────────────────
# 7. FULL TOPIC-DOC MATRIX (optional)
# ─────────────────────────────────────────
# W_df shows how much each doc belongs to each topic
W_df = pd.DataFrame(
    W,
    columns=[f"topic_{i}" for i in range(N_TOPICS)],
    index=df_post.index
)
W_df["title"] = df_post["title"].values
W_df["assigned_topic"] = doc_topics

print("\n" + "="*50)
print("TOPIC WEIGHTS PER DOCUMENT (first 10 rows)")
print("="*50)
print(W_df.head(10).to_string())