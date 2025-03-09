# LLMEvalSinhala
Utilizing RAG pipeline to enhance letter writing in Sinhala

## Description
This repo consists of code to process Sinhala official letters to be classified into groups (request letters, invitation letters, apology letters etc.) using OpenAI API with an engineered prompt.

Then the original Sinhala input and translated version to English will be saved as a CSV with metadata.

Also this consists of code to embed these documents into a vector space and create a FAISS index so that it can be utilized in a RAG pipeline for later use.

## Setup

### Directory Structure
Create the following directories in the root folder:
```
mkdir dataset processed_data embeddings
```

Then copy the source .txt files to the dataset directory.

### Prerequisites
- Python

### Installation
Install required libraries:
```
pip install pandas spacy scikit-learn transformers torch numpy openai
```

Download Spacy models:
```
# For Sinhala if available:
pip install https://github.com/explosion/spacy-models/releases/download/si_core_news_lg-3.0.0/si_core_news_lg-3.0.0.tar.gz
# Or multilingual model:
python -m spacy download xx_ent_wiki_sm
```

Set up OpenAI API key:
```
export OPENAI_API_KEY="your-api-key"
```

## Usage
Run the following command to process the dataset:
```
python letter_processor_enhanced.py
```
