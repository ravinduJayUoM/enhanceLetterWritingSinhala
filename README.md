# Sinhala Letter Generation System

This repository contains a comprehensive system for generating formal Sinhala letters using Retrieval-Augmented Generation (RAG) with custom information extraction.

## System Overview

The Sinhala Letter Generation System combines several components:
- Information extraction from user requests
- Vector database of high-quality Sinhala letter examples
- Retrieval-based prompt enhancement
- Fine-tunable NER for Sinhala-specific entity recognition
- API for integration with web applications

## Setup and Installation

### 1. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

### Quick Installation for Sinhala Letter RAG

To run the Sinhala letter RAG system specifically, you need to install the following packages:

```bash
rm -rf venv
rm -rf rag/faiss_index
```

```bash
pip install pandas langchain langchain-community langchain_openai torch fastapi uvicorn sentence-transformers faiss-cpu
```

You also need to set your OpenAI API key as an environment variable:

```bash
# For Linux/macOS
export OPENAI_API_KEY="your-api-key-here"

# For Windows (Command Prompt)
set OPENAI_API_KEY=your-api-key-here

# For Windows (PowerShell)
$env:OPENAI_API_KEY="your-api-key-here"
```

## Running the Letter Generation API

### 1. Build the Knowledge Base

First, build the vector database from the letter examples:

```bash
cd rag
python -c "from sinhala_letter_rag import letter_db; letter_db.build_knowledge_base(force_rebuild=True)"
```

Or use the API endpoint:
```bash
curl -X POST "http://localhost:8000/rebuild_knowledge_base/?force=true"
```

### 2. Start the API Server

```bash
cd rag
python sinhala_letter_rag.py
```

This will start a FastAPI server on http://localhost:8000

### 3. Using the API Endpoints

#### Process a Letter Request
```bash
curl -X POST "http://localhost:8000/process_query/" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "විදුහල්පතිතුමා වෙත, මගේ දරුවාට පාසල් නිවාඩු ඉල්ලීමක්"}'
```

If information is missing, the API will return questions to get additional details.

#### Generate the Final Letter
```bash
curl -X POST "http://localhost:8000/generate_letter/" \
  -H "Content-Type: application/json" \
  -d '{"original_prompt": "...", "enhanced_prompt": "..."}'
```

#### Search the Knowledge Base
```bash
curl "http://localhost:8000/search/?query=ඉල්ලීම&top_k=3"
```

#### Diagnostics
```bash
curl "http://localhost:8000/diagnostics/"
```

## Fine-tuning Custom Information Extraction

The system includes a custom NER (Named Entity Recognition) model for extracting structured information from Sinhala letter requests. You can fine-tune this model on your own data.

### 1. Prepare Datasets for Fine-tuning

The system automatically creates a labeled dataset from your existing letter examples:

```bash
python rag/models/prepare_ner_dataset.py
```

This script:
- Reads letters from the `dataset/` directory
- Automatically extracts entities using rule-based patterns
- Creates training and validation splits
- Saves the data in formats ready for fine-tuning

The prepared datasets will be stored in `rag/models/training_data/`.

### 2. Fine-tune the NER Model

```bash
python rag/finetune_ner_model.py
```

Additional options:
```bash
python rag/finetune_ner_model.py --epochs 5 --batch-size 16 --lr 3e-5 --model "xlm-roberta-base"
```

The fine-tuned model will be saved to `rag/models/training_data/best_model/`.

### 3. Test the Fine-tuned Model

Install dependencies

```bash
pip install pandas torch transformers spacy
```

To test how well the model extracts information from Sinhala text:

```bash
python rag/test_ner_model.py
```

This will show extraction results for example letters, comparing the performance of different extraction methods.

## Process Flow

When a letter generation is requested, the following process occurs:

1. **Initial Query Processing**:
   - User sends a query with their request in Sinhala
   - System handles this through the `RAGProcessor.process_query` method

2. **Information Extraction**:
   - Custom NER model extracts structured information (letter type, recipient, etc.)
   - LLM is used as a backup or to fill in missing information

3. **Missing Information Check**:
   - System identifies required fields based on letter type
   - If information is missing, generates questions in Sinhala
   - Returns these questions with status "incomplete"

4. **Information Retrieval**:
   - Once all information is available, retrieves relevant content
   - Searches the vector database using the extracted information
   - Returns the most relevant letter examples

5. **Enhanced Prompt Construction**:
   - Creates a detailed prompt combining the original request, extracted information, and relevant examples
   - Provides structure and formatting instructions

6. **Letter Generation**:
   - Uses ChatGPT or other LLM to generate the final letter
   - Returns the properly formatted Sinhala letter

## Troubleshooting

### Dependency Issues

If you encounter dependency conflicts:
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-deps
pip install the-problematic-package
```

### Database Issues

If the vector database fails to build:
```bash
rm -rf rag/chroma_db
python rag/sinhala_letter_rag.py
```

## Contributing

Contributions to improve the system are welcome. Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.