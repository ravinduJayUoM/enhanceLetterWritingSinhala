# sinhala_letter_rag.py
# Complete implementation for Sinhala letter RAG system

import os
import pandas as pd
import numpy as np
import time
import shutil
import stat
from typing import List, Dict, Any, Optional

# LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Import the NER model
from models.sinhala_ner import create_model

# API components
from fastapi import FastAPI, HTTPException, Query, Body
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Path configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "sinhala_letters.csv")
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")

# Model configuration
EMBEDDING_MODEL = "sentence-transformers/LaBSE"  # Good for Sinhala language support
LLM_MODEL = "gpt-4"  # You can replace with your preferred model

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"}  # Use "cuda" if you have GPU support
)

def ensure_directory_writable(dir_path: str):
    """Ensure a directory is writable, fixing permissions if necessary."""
    if os.path.exists(dir_path):
        try:
            # Test write permissions by creating a temporary file
            test_file = os.path.join(dir_path, ".permission_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            print(f"Directory {dir_path} is writable")
        except (PermissionError, OSError) as e:
            print(f"Permission issue detected with {dir_path}: {str(e)}")
            print("Attempting to fix permissions...")
            try:
                # Change permissions recursively to make the directory writable
                for root, dirs, files in os.walk(dir_path):
                    os.chmod(root, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 0777
                    for d in dirs:
                        os.chmod(os.path.join(root, d), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                    for f in files:
                        os.chmod(os.path.join(root, f), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                print(f"Permissions fixed for {dir_path}")
            except Exception as fix_error:
                print(f"Failed to fix permissions for {dir_path}: {str(fix_error)}")
                raise PermissionError(f"Cannot make {dir_path} writable. Please fix manually.")
    else:
        # Create the directory if it doesn't exist
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created directory {dir_path}")
        except Exception as e:
            print(f"Failed to create directory {dir_path}: {str(e)}")
            raise

class LetterDatabase:
    def __init__(self, csv_path=DATA_PATH, persist_dir=CHROMA_PERSIST_DIR):
        """Initialize the letter database."""
        self.csv_path = csv_path
        self.persist_dir = persist_dir
        self.db = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]  # Customize for Sinhala
        )
    
    def load_data(self) -> pd.DataFrame:
        """Load letter data from CSV file."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found at {self.csv_path}")
        
        df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(df)} letters from CSV")
        return df
    
    def create_documents(self, df: pd.DataFrame) -> List[Document]:
        """Create documents from dataframe with proper metadata."""
        documents = []
        
        for _, row in df.iterrows():
            # Extract subject, content and tags
            subject = row['subject']
            content = row['content']
            tags = row['tags']
            print(f"Processing row: subject={subject}, content={content}, tags={tags}")
            
            # Create the full text combining subject and content
            text = f"{subject}\n\n{content}"
            
            # Create metadata with tags as string (not list)
            metadata = {
                "subject": subject,
                "tags": tags,  # Store as string to avoid Chroma error
                "source": "sinhala_letters_dataset"
            }
            
            # Add to documents
            documents.append(Document(page_content=text, metadata=metadata))
        
        print(f"Created {len(documents)} documents with string metadata (no lists)")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks for better retrieval."""
        split_docs = self.text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
        return split_docs
    
    def create_vectorstore(self, documents: List[Document], store_type="faiss"):  # Default to FAISS
        """Create and return a vector store with the provided documents."""
        if not documents:
            print("WARNING: No documents provided to create vector store!")
            return None
            
        print(f"Creating {store_type} vector store with {len(documents)} documents")
        start_time = time.time()
        
        if store_type.lower() == "chroma":
            # Create Chroma vector store
            if not os.path.exists(self.persist_dir):
                os.makedirs(self.persist_dir, exist_ok=True)
                
            # Ensure the directory is writable
            ensure_directory_writable(self.persist_dir)
                
            print(f"Starting embedding generation for {len(documents)} documents...")
            
            try:
                db = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    persist_directory=self.persist_dir
                )
                db.persist()
                self.db = db
                
                # Get document count after creation to verify
                doc_count = self.get_document_count() if self.db else 0
                elapsed_time = time.time() - start_time
                
                print(f"Vector store creation completed in {elapsed_time:.2f} seconds")
                print(f"Created Chroma vector store at {self.persist_dir} with {doc_count} documents")
            
            except Exception as e:
                print(f"ERROR creating Chroma vector store: {str(e)}")
                if "readonly database" in str(e).lower():
                    print("This appears to be a permission issue with the database files.")
                    print(f"Please check that the directory {self.persist_dir} is writable.")
                    print("You may need to run: chmod -R 777 " + self.persist_dir)
                raise
            
        elif store_type.lower() == "faiss":
            # Create FAISS vector store
            faiss_path = os.path.join(BASE_DIR, "faiss_index")
            db = FAISS.from_documents(documents, embeddings)
            
            # Save the FAISS index
            if not os.path.exists(faiss_path):
                os.makedirs(faiss_path, exist_ok=True)
            db.save_local(faiss_path)
            self.db = db
            print(f"Created FAISS vector store at {faiss_path}")
            
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")
        
        return self.db
    
    def load_vectorstore(self, store_type="faiss"):  # Default to FAISS
        """Load existing vector store if available."""
        if store_type.lower() == "chroma":
            if os.path.exists(self.persist_dir):
                self.db = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=embeddings,
                    allow_dangerous_deserialization=True  # Add this parameter to allow pickle deserialization
                )
                print(f"Loaded existing Chroma vector store from {self.persist_dir}")
                return self.db
        
        elif store_type.lower() == "faiss":
            faiss_path = os.path.join(BASE_DIR, "faiss_index")
            if os.path.exists(faiss_path):
                self.db = FAISS.load_local(
                    faiss_path,
                    embeddings,
                    allow_dangerous_deserialization=True  # Allow pickle deserialization for trusted local index
                )
                print(f"Loaded existing FAISS vector store from {faiss_path}")
                return self.db
        
        return None
    
    def build_knowledge_base(self, store_type="faiss", force_rebuild=False):  # Default to FAISS
        """Build the knowledge base from the dataset."""
        # Try to load existing vector store if not forcing a rebuild
        if not force_rebuild:
            existing_db = self.load_vectorstore(store_type)
            if existing_db is not None:
                return existing_db
        
        # If forcing rebuild or no existing DB found, create a new one
        print("Building new knowledge base...")
        df = self.load_data()
        documents = self.create_documents(df)
        split_docs = self.split_documents(documents)
        db = self.create_vectorstore(split_docs, store_type)
        return db
    
    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """Search the vector store for relevant documents."""
        if self.db is None:
            raise ValueError("Database not initialized. Call build_knowledge_base first.")
        
        # Print debug info to help diagnose issues
        print(f"Searching for: '{query}' with top_k={top_k}")
        
        # Get the embedding for the query to see if it's working
        try:
            query_embedding = embeddings.embed_query(query)
            print(f"Query embedding generated successfully: {len(query_embedding)} dimensions")
        except Exception as e:
            print(f"Error generating query embedding: {str(e)}")
            raise
        
        results = self.db.similarity_search(query, k=top_k)
        print(f"Found {len(results)} results")
        return results
    
    def get_document_count(self) -> int:
        """Get the number of documents in the vector store."""
        if self.db is None:
            raise ValueError("Database not initialized. Call build_knowledge_base first.")
        
        # This implementation depends on which vector store is used
        if isinstance(self.db, Chroma):
            return self.db._collection.count()
        elif isinstance(self.db, FAISS):
            # For FAISS, we can get the index size
            return len(self.db.index_to_docstore_id)
        else:
            return -1  # Unknown vector store type
            
    def get_sample_documents(self, count: int = 3) -> List[Dict[str, Any]]:
        """Get sample documents from the vector store."""
        if self.db is None:
            raise ValueError("Database not initialized. Call build_knowledge_base first.")
        
        if isinstance(self.db, Chroma):
            # For Chroma, we can get a sample of documents
            docs = self.db._collection.get(limit=count)
            return [
                {"id": id, "text": docs["documents"][i], "metadata": docs["metadatas"][i]}
                for i, id in enumerate(docs["ids"][:count])
            ]
        elif isinstance(self.db, FAISS):
            # For FAISS, we can get a sample of documents
            sample_ids = list(self.db.index_to_docstore_id.values())[:count]
            return [
                {"id": id, "text": self.db.docstore.search(id).page_content, 
                 "metadata": self.db.docstore.search(id).metadata}
                for id in sample_ids
            ]
        else:
            return []  # Unknown vector store type

class UserQuery(BaseModel):
    """Pydantic model for user query."""
    prompt: str
    missing_info: Optional[Dict[str, str]] = None

class LetterRequest(BaseModel):
    """Pydantic model for letter generation request."""
    original_prompt: str
    enhanced_prompt: str

class RAGProcessor:
    def __init__(self, letter_db: LetterDatabase):
        """Initialize the RAG processor."""
        self.letter_db = letter_db
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1)
        
        # Load the NER model correctly first
        try:
            # Create the base NER model
            self.ner_model = create_model(
                model_name="xlm-roberta-base",  # Base model used during fine-tuning
                use_spacy=True,
                use_rules=True
            )
            
            # Path to the fine-tuned model
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                "models/training_data/best_model"
            )
            
            # Load the fine-tuned model weights if they exist
            if os.path.exists(model_path):
                try:
                    from transformers import AutoModelForTokenClassification, AutoTokenizer
                    # Update the model's transformer components with fine-tuned weights
                    self.ner_model.model = AutoModelForTokenClassification.from_pretrained(model_path)
                    self.ner_model.tokenizer = AutoTokenizer.from_pretrained(model_path)
                    print(f"Successfully loaded fine-tuned NER model from {model_path}")
                except Exception as e:
                    print(f"Error loading fine-tuned model weights: {str(e)}")
                    print("Using base model without fine-tuned weights")
            else:
                print(f"Fine-tuned model not found at {model_path}. Using base model.")
                
            print("NER model loaded successfully")
        except Exception as e:
            print(f"Error loading NER model: {str(e)}")
            self.ner_model = None
    
    def extract_key_info(self, prompt: str) -> Dict[str, Any]:
        """Extract key information from the user prompt using the fine-tuned NER model."""
        try:
            # Use the pre-loaded NER model instance
            if self.ner_model is None:
                print("NER model was not loaded during initialization, falling back to LLM extraction")
                return self._extract_with_llm(prompt)
            
            # Use the NER model to extract entities - using extract_info method instead of extract_entities
            extracted_info = self.ner_model.extract_info(prompt)
            
            # Map NER model output to our expected format if needed
            # The NER model already returns a dictionary with appropriate keys
            
            # If the NER model doesn't extract all needed information,
            # use the LLM as a backup for missing fields
            missing_fields = [k for k, v in extracted_info.items() if not v]
            if missing_fields:
                print("NER extracted partial information. Using LLM as backup for missing fields.")
                llm_extraction = self._extract_with_llm(prompt)
                
                # Merge NER and LLM results, prioritizing NER results
                for key in missing_fields:
                    if key in llm_extraction and llm_extraction.get(key):
                        extracted_info[key] = llm_extraction.get(key, "")
            
            return extracted_info
            
        except Exception as e:
            print(f"Error using NER model: {str(e)}")
            print("Falling back to LLM-based extraction")
            # Fall back to LLM-based extraction if the NER model fails
            return self._extract_with_llm(prompt)
    
    def _extract_with_llm(self, prompt: str) -> Dict[str, Any]:
        """Extract key information from the user prompt using LLM."""
        # Use LLM to extract structured information from Sinhala prompt (backup method)
        extraction_prompt = ChatPromptTemplate.from_template("""
        මෙම ලිපි ඉල්ලීමෙන් ප්‍රධාන තොරතුරු උපුටා ගන්න. JSON ආකෘතියකින් පිළිතුරු දෙන්න:

        ඉල්ලීම:
        {prompt}

        JSON ආකෘතිය:
        {{
          "letter_type": "ලිපි වර්ගය (application/request/complaint/etc)",
          "recipient": "ලිපිය ලබන්නා",
          "sender": "ලිපිය යවන්නා",
          "subject": "ලිපියේ මාතෘකාව",
          "purpose": "අරමුණ",
          "details": "අමතර විස්තර",
          "missing_info": "අතුරුදහන් වූ වැදගත් තොරතුරු"
        }}
        """)
        
        extraction_chain = extraction_prompt | self.llm
        try:
            result = extraction_chain.invoke({"prompt": prompt})
            # Extract JSON from the response
            import json
            import re
            
            # Handle potential formatting issues in the LLM response
            json_match = re.search(r'(\{.*\})', result.content, re.DOTALL)
            if (json_match):
                json_str = json_match.group(1)
                extracted_info = json.loads(json_str)
                return extracted_info
            else:
                return {"error": "Failed to parse extraction result"}
        except Exception as e:
            print(f"Error extracting information: {str(e)}")
            return {"error": str(e)}
    
    def identify_missing_info(self, extracted_info: Dict[str, Any]) -> List[str]:
        """Identify missing information based on letter type."""
        letter_type = extracted_info.get("letter_type", "general")
        missing_info = []
        
        # Basic required fields for all letter types
        required_fields = ["recipient", "sender", "subject", "purpose"]
        
        # Add letter-type specific required fields
        if letter_type == "application":
            required_fields.extend(["qualifications", "contact_details"])
        elif letter_type == "complaint":
            required_fields.extend(["incident_date", "requested_action"])
        elif letter_type == "request":
            required_fields.extend(["requested_items", "timeline"])
        
        # Check which fields are missing or empty
        for field in required_fields:
            value = extracted_info.get(field, "")
            if not value or value.strip() == "" or value == "N/A":
                missing_info.append(field)
        
        return missing_info
    
    def generate_questions(self, missing_fields: List[str]) -> Dict[str, str]:
        """Generate questions in Sinhala for missing information."""
        # Map missing fields to Sinhala questions
        question_mapping = {
            "recipient": "ලිපිය යොමු කළ යුත්තේ කාටද?",  # Who should the letter be addressed to?
            "sender": "ලිපිය යවන්නේ කවුරුන්ද?",  # Who is sending the letter?
            "subject": "ලිපියේ මාතෘකාව කුමක්ද?",  # What is the subject of the letter?
            "purpose": "ලිපියේ මූලික අරමුණ කුමක්ද?",  # What is the main purpose of the letter?
            "qualifications": "ඔබගේ සුදුසුකම් මොනවාද?",  # What are your qualifications?
            "contact_details": "ඔබගේ සම්බන්ධතා විස්තර මොනවාද?",  # What are your contact details?
            "incident_date": "සිද්ධිය සිදු වූයේ කවදාද?",  # When did the incident occur?
            "requested_action": "ඔබ ඉල්ලා සිටින ක්‍රියාමාර්ගය කුමක්ද?",  # What action are you requesting?
            "requested_items": "ඔබ ඉල්ලුම් කරන දේ මොනවාද?",  # What are you requesting?
            "timeline": "මෙය අවශ්‍ය කාල රාමුව කුමක්ද?",  # What is the timeframe needed?
        }
        
        questions = {}
        for field in missing_fields:
            if field in question_mapping:
                questions[field] = question_mapping[field]
            else:
                questions[field] = f"කරුණාකර {field} සපයන්න"  # Please provide {field}
        
        return questions
    
    def retrieve_relevant_content(self, info: Dict[str, Any], top_k: int = 3) -> List[Document]:
        """Retrieve relevant content from the knowledge base."""
        # Construct a search query from the extracted information
        letter_type = info.get("letter_type", "")
        subject = info.get("subject", "")
        purpose = info.get("purpose", "")
        details = info.get("details", "")
        
        # Create a search query combining the key information
        search_query = f"{letter_type} {subject} {purpose} {details}"
        
        # Search the vector store
        results = self.letter_db.search(search_query, top_k=top_k)
        return results
    
    def construct_enhanced_prompt(
        self, 
        original_prompt: str,
        extracted_info: Dict[str, Any],
        retrieved_examples: List[Document],
        missing_info_answers: Optional[Dict[str, str]] = None
    ) -> str:
        """Construct an enhanced prompt for the LLM to generate the letter."""
        # Merge extracted info with answers to missing info questions
        complete_info = extracted_info.copy()
        if missing_info_answers:
            for key, value in missing_info_answers.items():
                complete_info[key] = value
        
        # Extract letter structure and examples from retrieved documents
        example_texts = []
        for doc in retrieved_examples:
            example_texts.append(doc.page_content)
        
        examples_str = "\n\n---\n\n".join(example_texts)
        
        # Construct the enhanced prompt
        enhanced_prompt = f"""
        මෙම ලිපිය සිංහලෙන් ලියන්න. මුල් ඉල්ලීම: {original_prompt}
        
        අවශ්‍ය ලිපි වර්ගය: {complete_info.get('letter_type', 'general')}
        
        ලිපිය ලබන්නා: {complete_info.get('recipient', '')}
        ලිපිය යවන්නා: {complete_info.get('sender', '')}
        මාතෘකාව: {complete_info.get('subject', '')}
        අරමුණ: {complete_info.get('purpose', '')}
        
        අමතර විස්තර:
        {complete_info.get('details', '')}
        
        # ලිපි ආකෘති උදාහරණ (නව ලිපියේ මෙම පෙළ ඇතුළත් නොකරන්න, ඔබට ආකෘතිය පමණක් අනුගමනය කරන්න):
        {examples_str}
        
        ලිපිය නිවැරදි සිංහල ව්‍යාකරණ, විරාම ලකුණු සහ ව්‍යවහාරික විධිමත් භාෂාව භාවිතා කර ලියන්න. විධිමත් ලිපියකට සුදුසු ආදර වචන භාවිතා කරන්න. අවසාන ලිපිය අන්තර්ගතය පමණක් ලබා දෙන්න, කිසිදු අමතර පැහැදිලි කිරීමක් හෝ සටහන් නොමැතිව.
        """
        
        return enhanced_prompt
    
    def process_query(self, user_query: UserQuery) -> Dict[str, Any]:
        """Process a user query and identify missing information."""
        # Extract information from the prompt
        extracted_info = self.extract_key_info(user_query.prompt)
        
        # Identify missing information
        missing_fields = self.identify_missing_info(extracted_info)
        
        if (missing_fields and not user_query.missing_info):
            # If missing info and no answers provided, return questions
            questions = self.generate_questions(missing_fields)
            return {
                "status": "incomplete",
                "extracted_info": extracted_info,
                "missing_fields": missing_fields,
                "questions": questions
            }
        
        # If all info is present or answers provided, retrieve relevant content
        relevant_content = self.retrieve_relevant_content(extracted_info)
        
        # Construct enhanced prompt
        enhanced_prompt = self.construct_enhanced_prompt(
            user_query.prompt,
            extracted_info,
            relevant_content,
            user_query.missing_info
        )
        
        return {
            "status": "complete",
            "extracted_info": extracted_info,
            "enhanced_prompt": enhanced_prompt,
            "relevant_docs": [doc.page_content for doc in relevant_content]
        }

# Initialize FastAPI
app = FastAPI(title="Sinhala Letter RAG System")

# Setup CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize letter database and RAG processor
letter_db = LetterDatabase()
rag_processor = None

@app.on_event("startup")
async def startup_event():
    """Initialize the knowledge base on startup."""
    global rag_processor
    try:
        db = letter_db.build_knowledge_base()
        rag_processor = RAGProcessor(letter_db)
        print("RAG processor initialized successfully")
    except Exception as e:
        print(f"Error initializing knowledge base: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {"status": "Sinhala Letter RAG System is running"}

@app.post("/process_query/")
async def process_query(query: UserQuery):
    """Process a user query and identify missing information."""
    if rag_processor is None:
        raise HTTPException(status_code=500, detail="RAG processor not initialized")
    
    try:
        result = rag_processor.process_query(query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_letter/")
async def generate_letter(request: LetterRequest):
    """Generate a letter based on the enhanced prompt."""
    if rag_processor is None:
        raise HTTPException(status_code=500, detail="RAG processor not initialized")
    
    try:
        # Generate the letter using the LLM
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0.3)
        letter_prompt = ChatPromptTemplate.from_template("{enhanced_prompt}")
        letter_chain = letter_prompt | llm | StrOutputParser()
        
        letter = letter_chain.invoke({"enhanced_prompt": request.enhanced_prompt})
        
        return {"generated_letter": letter}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/")
async def search_kb(query: str = Query(...), top_k: int = Query(3)):
    """Search the knowledge base directly."""
    if letter_db.db is None:
        raise HTTPException(status_code=500, detail="Knowledge base not initialized")
    
    try:
        results = letter_db.search(query, top_k=top_k)
        return {
            "query": query,
            "top_k": top_k,
            "result_count": len(results),
            "results": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in results
            ]
        }
    except Exception as e:
        print(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rebuild_knowledge_base/")
async def rebuild_knowledge_base(force: bool = Query(True)):
    """Rebuild the knowledge base from scratch."""
    global rag_processor
    
    try:
        # Check if the csv file exists
        if not os.path.exists(DATA_PATH):
            raise HTTPException(status_code=404, detail=f"Data file not found: {DATA_PATH}")
        
        # Log the start of the rebuild process
        print(f"Starting knowledge base rebuild process. Force rebuild: {force}")
        
        # Delete existing vector store if it exists
        if os.path.exists(CHROMA_PERSIST_DIR) and force:
            try:
                print(f"Removing existing vector store at {CHROMA_PERSIST_DIR}")
                # First, ensure the directory is writable
                ensure_directory_writable(CHROMA_PERSIST_DIR)
                
                # Try to delete all files individually first to avoid permission issues
                for root, dirs, files in os.walk(CHROMA_PERSIST_DIR, topdown=False):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            os.remove(file_path)
                            print(f"Removed file: {file_path}")
                        except Exception as e:
                            print(f"Failed to remove file {file_path}: {str(e)}")
                    
                    for dir in dirs:
                        dir_path = os.path.join(root, dir)
                        try:
                            shutil.rmtree(dir_path)
                            print(f"Removed directory: {dir_path}")
                        except Exception as e:
                            print(f"Failed to remove directory {dir_path}: {str(e)}")
                
                # Now try to remove the main directory
                shutil.rmtree(CHROMA_PERSIST_DIR)
                print("Existing vector store removed")
            except Exception as e:
                print(f"Error removing vector store: {str(e)}")
                # If removal fails, try creating a new directory with a timestamp suffix
                new_dir = f"{CHROMA_PERSIST_DIR}_{int(time.time())}"
                os.makedirs(new_dir, exist_ok=True)
                print(f"Created new vector store directory: {new_dir}")
                letter_db.persist_dir = new_dir
                
        # Ensure the vector store directory exists and is writable
        if not os.path.exists(CHROMA_PERSIST_DIR):
            os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        ensure_directory_writable(CHROMA_PERSIST_DIR)
        
        # Load the data
        start_time = time.time()
        print("Loading data from CSV...")
        df = letter_db.load_data()
        
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="CSV file contains no data")
        
        # Create documents
        print(f"Creating documents from {len(df)} rows...")
        documents = letter_db.create_documents(df)
        print(f"Created {len(documents)} document objects")
        
        # Split documents
        print("Splitting documents into chunks...")
        split_docs = letter_db.split_documents(documents)
        print(f"Split into {len(split_docs)} document chunks")
        
        # Create vector store
        print("Creating vector store (this may take some time)...")
        db = letter_db.create_vectorstore(split_docs, "chroma")
        
        # Reinitialize the RAG processor
        rag_processor = RAGProcessor(letter_db)
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Verify document count
        doc_count = letter_db.get_document_count()
        
        return {
            "status": "success",
            "message": "Knowledge base rebuilt successfully",
            "original_data_rows": len(df),
            "documents_created": len(documents),
            "document_chunks": len(split_docs),
            "documents_in_vector_store": doc_count,
            "time_taken_seconds": round(total_time, 2),
            "vector_store_path": letter_db.persist_dir
        }
    
    except Exception as e:
        print(f"Error rebuilding knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/diagnostics/")
async def run_diagnostics():
    """Run diagnostics on the knowledge base."""
    if letter_db.db is None:
        raise HTTPException(status_code=500, detail="Knowledge base not initialized")
    
    try:
        document_count = letter_db.get_document_count()
        sample_documents = letter_db.get_sample_documents(3)
        
        # Try a simple search to see if it returns anything
        test_query = "application"
        test_results = letter_db.search(test_query, top_k=1)
        
        # Check CSV file existence and row count
        csv_exists = os.path.exists(DATA_PATH)
        csv_row_count = 0
        if csv_exists:
            try:
                df = pd.read_csv(DATA_PATH)
                csv_row_count = len(df)
            except Exception as e:
                print(f"Error reading CSV: {str(e)}")
        
        return {
            "status": "Knowledge base is initialized",
            "document_count": document_count,
            "embedding_model": EMBEDDING_MODEL,
            "sample_documents": sample_documents,
            "test_search": {
                "query": test_query,
                "results_found": len(test_results) > 0,
                "first_result": test_results[0].page_content if test_results else None
            },
            "data_source": {
                "csv_exists": csv_exists,
                "csv_path": DATA_PATH,
                "csv_row_count": csv_row_count
            }
        }
    except Exception as e:
        print(f"Diagnostics error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run("sinhala_letter_rag:app", host="0.0.0.0", port=8000, reload=True)