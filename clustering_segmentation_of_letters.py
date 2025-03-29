import os
import re
import glob
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
import nltk
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---- CONFIG ----
LETTER_DIR = '/home/ravindu_23/letters/'  # Path to your folder
NUM_CLUSTERS = 10  # Adjust based on expected letter types
NUM_LETTER_CLUSTERS = 5 # Adjust based on expected letter sections

# ---- HELPERS ----

def clean_text(text):
    """ Normalize and clean raw Sinhala text. """
    text = re.sub(r'[^\u0D80-\u0DFF\s.,;:\-–—!?()\[\]{}"\']+', '', text)  # Keep Sinhala characters and common punctuation
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()

def simple_sentence_segment(text):
    """Simple but robust sentence segmentation based on punctuation."""
    # Split on common sentence ending punctuation
    sentences = re.split(r'[.!?]+', text)
    # Remove empty sentences and strip whitespace
    return [s.strip() for s in sentences if s.strip()]

def cluster_letter(letter_sentences, num_clusters=NUM_LETTER_CLUSTERS):
    """ Cluster the sentences in the letter using TF-IDF and KMeans. """
    if not letter_sentences:
        logging.warning("Empty letter sentences list")
        return {}
    
    # Filter out very short sentences that won't contribute to clustering
    filtered_sentences = [s for s in letter_sentences if len(s.split()) >= 3]
    
    if not filtered_sentences:
        logging.warning("No sentences long enough for clustering")
        return {}
    
    try:
        # Try using TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=500,
            min_df=1,  # Accept terms that appear in at least 1 document
            analyzer='char_wb',  # Character n-grams, including word boundaries
            ngram_range=(2, 5)  # Use 2-5 character n-grams to capture Sinhala patterns
        )
        X = vectorizer.fit_transform(filtered_sentences)
        
        if X.shape[1] == 0:
            raise ValueError("Empty vocabulary from TF-IDF")
        
    except Exception as e:
        logging.warning(f"TF-IDF vectorization failed: {e}. Trying count vectorizer.")
        try:
            # Fall back to simple count vectorizer
            vectorizer = CountVectorizer(
                analyzer='char_wb',
                ngram_range=(2, 5)
            )
            X = vectorizer.fit_transform(filtered_sentences)
            
            if X.shape[1] == 0:
                logging.warning("Both vectorizers failed to produce features.")
                return {}
                
        except Exception as e2:
            logging.error(f"Count vectorization also failed: {e2}")
            return {}
    
    # Adjust number of clusters based on available data
    effective_clusters = min(num_clusters, len(filtered_sentences))
    if effective_clusters <= 1:
        return {"segment_1": " ".join(filtered_sentences)}
    
    try:
        kmeans = MiniBatchKMeans(
            n_clusters=effective_clusters, 
            random_state=42,
            n_init=10,
            batch_size=256
        )
        clusters = kmeans.fit_predict(X)
        
        segments = {}
        for i in range(effective_clusters):
            segment_sentences = [filtered_sentences[j] for j in range(len(filtered_sentences)) if clusters[j] == i]
            if segment_sentences:  # Only add non-empty segments
                segments[f'segment_{i+1}'] = ' '.join(segment_sentences)
        
        return segments
        
    except Exception as e:
        logging.error(f"Clustering failed: {e}")
        return {"segment_1": " ".join(filtered_sentences)}

# ---- MAIN SCRIPT ----

all_letters = []
file_paths = sorted(glob.glob(os.path.join(LETTER_DIR, '*.txt')))

print(f"Found {len(file_paths)} letter files.")

for file_path in tqdm(file_paths, desc="Processing letters"):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw = f.read()
    except UnicodeDecodeError:
        print(f"⚠️ Skipping unreadable file: {file_path}")
        continue

    cleaned = clean_text(raw)
    sentences = simple_sentence_segment(cleaned)  # Use simpler sentence segmentation

    # Cluster the entire letter
    letter_clusters = cluster_letter(sentences)

    all_segments = {
        'file': os.path.basename(file_path),
        'full_text': cleaned,
        **letter_clusters
    }

    all_letters.append(all_segments)

df = pd.DataFrame(all_letters)
df.to_csv('sinhala_letters_segmented.csv', index=False)
print("✅ Saved segmented letters to sinhala_letters_segmented.csv")

