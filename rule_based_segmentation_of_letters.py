import os
import re
import glob
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import sent_tokenize

# ---- CONFIG ----
LETTER_DIR = '/Users/mihiranga/Msc/research-code/dataset'  # Path to your folder
NUM_CLUSTERS = 10  # Adjust based on expected letter types

# ---- HELPERS ----

def clean_text(text):
    """ Normalize and clean raw Sinhala text. """
    text = re.sub(r'[^\u0D80-\u0DFF\s.,;:\-–—!?()\[\]{}"\']+', '', text)  # Keep Sinhala characters and common punctuation
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()

def sentence_segment(text):
    """ Naive sentence splitter for Sinhala (can be improved with tokenizers later). """
    # Basic segmentation using punctuation
    return re.split(r'(?<=[\u0D9A-\u0DC6][.!?])\s+', text)

def find_segments(sentences):
    """ Identify potential segments based on known formal patterns. """
    categories = {
        'greeting': ['ආදරණීය', 'මහත්මයාණනි', 'මහත්මිය', 'ආයුබෝවන්'],
        'request': ['කරුණාකර', 'කටයුතු කරන්න', 'අනුමැතිය ඉල්ලා', 'දැනුවත් කරන මෙන්', 'කාරුණිකව දන්වමි', 'මා වෙත', 'අයදුම්පත්‍රය'],
        'acknowledgment': ['සතුටුයි', 'ස්ථිර කරනු', 'කඩිනම්', 'හා බැන්දේ', 'වැඩිමනත්වයි', 'අමුණා', 'සහතික කරන ලද', 'දැනුම් දී ඇත'],
        'regret': ['කණගාටුයි', 'සමාවන්න', 'අපහසුතාවයට', 'ආපදා', 'බාධාවක්', 'අඩපණ', 'අපහසුතාවයකි'],
        'gratitude': ['ස්තුතියි', 'ආදරයෙන්', 'අගේ කොට', 'දීමනා'],
        'complaint': ['පැමිණිලි', 'පැමිණිල්ලේ'],
        'invitation': ['ආරාධනා', 'ආරාධනායි', 'ආරාධනා කරන', 'ආරාධනා කරමි'],
        'confirmation': ['සනාථ', 'සනාථ කර', 'පත්වීම', 'වෙන්කර ගැනීම'],
        'information': ['තොරතුරු', 'පමණක්', 'වෙන් කර ඇත', 'උක්ත කරුණ', 'උපදෙස්', 'කොන්දේසී', 'කරුණු'],
        'warning': ['අවවාද', 'කළ යුතුය', 'ලැබිය යුතුය', 'අපොහොසත් වන', 'මාරක', 'අනතුරු', 'අවදානම්'],
        'agreement': ['එකඟ වී', 'සහයෝගය ලබා', 'සාකච්ඡා කිරීම', 'ශක්‍යතාවය', 'කල හැකිය', 'අවසරය'],
        'disagreement': ['නොමැත', 'නොමැති', 'අසමඟයි', 'අප්‍රසාදයට', 'නොසලකන', 'අවලංගු'],
        'notification': ['දැනුම්දීම', 'සම්පුර්ණ කළ යුතු', 'දැනුවත් කරනු', 'කටයුතු යොදා ඇත', 'දිනට පෙර'],
        'closing': ['ස්තුතියි මෙයට', 'ඔබගේ'],
        'order': ['නියම කර', 'වාර්තා කරන්න', 'අණ පරිදි', 'කොන්දේසි', 'අත්හිටවයි']
    }

    segments = {category: [] for category in categories}

    for s in sentences:
        matched = False
        for category, keywords in categories.items():
            if any(keyword in s for keyword in keywords):
                segments[category].append(s.strip())  # Add the sentence to the matched category
                matched = True
                break
        if not matched:
            segments['information'].append(s.strip())  # Default to 'information' if no match

    # Combine sentences for each category
    return {category: ' '.join(sentences) for category, sentences in segments.items()}

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
    sentences = sentence_segment(cleaned)
    segments = find_segments(sentences)

    all_letters.append({
        'file': os.path.basename(file_path),
        'full_text': cleaned,
        'greeting': segments['greeting'],
        'request': segments['request'],
        'acknowledgment': segments['acknowledgment'],
        'regret': segments['regret'],
        'gratitude': segments['gratitude'],
        'complaint': segments['complaint'],
        'invitation': segments['invitation'],
        'confirmation': segments['confirmation'],
        'information': segments['information'],
        'warning': segments['warning'],
        'agreement': segments['agreement'],
        'disagreement': segments['disagreement'],
        'notification': segments['notification'],
        'closing': segments['closing'],
        'order': segments['order']
    })

df = pd.DataFrame(all_letters)
df.to_csv('sinhala_letters_preprocessed_1.csv', index=False)
print("✅ Saved preprocessed letters to sinhala_letters_preprocessed.csv")

# # ---- FREQUENT PHRASE EXTRACTION ----

# all_sentences = []
# for body in df['body']:
#     all_sentences.extend(sentence_segment(body))

# cleaned_sentences = [s.strip() for s in all_sentences if len(s.strip()) > 10]
# phrase_counts = Counter(cleaned_sentences)
# top_phrases = phrase_counts.most_common(50)

# # Save common segments
# with open('common_segments.txt', 'w', encoding='utf-8') as f:
#     for phrase, count in top_phrases:
#         f.write(f"{phrase} ## {count}\n")

# print("✅ Saved most common body phrases to common_segments.txt")

# # ---- CLUSTER LETTERS BY TF-IDF ----

# vectorizer = TfidfVectorizer(max_features=1000)
# X = vectorizer.fit_transform(df['full_text'])

# kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
# df['cluster'] = kmeans.fit_predict(X)

# df.to_csv('sinhala_letters_clustered_1.csv', index=False)
# print("✅ Letters clustered and saved to sinhala_letters_clustered.csv")

