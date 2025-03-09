import pandas as pd
import os
from pathlib import Path
import re
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import faiss
import torch
import numpy as np
from typing import Dict, List, Optional
from letter_translator import LetterTranslator

# Try to load Sinhala language model if available, or use a multilingual model
try:
    nlp = spacy.load("si_core_news_lg")
except:
    print("Sinhala spaCy model not found, using multilingual model instead")
    nlp = spacy.load("xx_ent_wiki_sm")

class LetterProcessor:
    def __init__(self, data_path: str = None, knowledge_base_path: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the letter processor
        
        Args:
            data_path: Path to letter dataset
            knowledge_base_path: Path to knowledge base resources
            api_key: OpenAI API key for translation service
        """
        self.data_path = Path(data_path)
        self.knowledge_base_path = Path(knowledge_base_path) if knowledge_base_path else None
        self.letter_df = None
        self.letter_types = []
        self.letter_embeddings = {}
        
        print("Initializing pre process")
        # Initialize translator
        self.translator = LetterTranslator(api_key=api_key)
        
        # Use SentenceTransformer with LaBSE model for better multilingual embeddings
        self.model = SentenceTransformer('sentence-transformers/LaBSE')
        self.faiss_index = None
        
    def process_dataset(self) -> pd.DataFrame:
        """Process and structure the letter dataset"""
        count = 1
        letters_data = []
        documents = []
        
        # Process each letter file
        for letter_file in self.data_path.glob("*.txt"):
            try:
                # Read file with explicit UTF-8 encoding for Sinhala
                try:
                    with open(letter_file, "r", encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    # If UTF-8 fails, try auto-detection with chardet
                    import chardet
                    with open(letter_file, "rb") as f:
                        raw_data = f.read()
                        encoding_result = chardet.detect(raw_data)
                        encoding = encoding_result['encoding']
                        content = raw_data.decode(encoding)
                
                documents.append(content)
                
                # Translate and classify the letter
                translation, letter_type, metadata = self.translator.translate_and_classify(content)
                
                print(f"File processed {count} {letter_file.name} with type {letter_type}")
                
                letters_data.append({
                    "file_name": letter_file.name,
                    "content": content,
                    "translation": translation,
                    "letter_type": letter_type,
                    "classification_confidence": metadata.get("confidence", "low"),
                    "classification_keywords": metadata.get("keywords_found", []),
                })
                count += 1
            except Exception as e:
                print(f"Error processing file {letter_file.name}: {str(e)}")
                continue
        
        if not letters_data:
            print("No files were processed successfully.")
            return pd.DataFrame()
            
        # Generate embeddings for all documents at once (more efficient)
        embeddings = self._create_embeddings(documents)
        
        # Add embeddings to the dataframe
        for i, data in enumerate(letters_data):
            data["embedding"] = embeddings[i]
        
        self.letter_df = pd.DataFrame(letters_data)
        self.letter_types = self.letter_df["letter_type"].unique().tolist()
        
        # Create FAISS index
        self._create_faiss_index(embeddings)
        
        return self.letter_df
    
    def _create_embeddings(self, documents: List[str]) -> np.ndarray:
        """Create vector embeddings for text content
        
        Args:
            documents: List of text documents to embed
            
        Returns:
            numpy array containing the embedding vectors
        """
        # Generate embeddings for documents
        document_embeddings = self.model.encode(documents, convert_to_tensor=True)
        # Normalize for cosine similarity
        doc_embeddings = normalize(document_embeddings.cpu().numpy(), axis=1)
        return doc_embeddings

    def _create_faiss_index(self, embeddings: np.ndarray) -> None:
        """Create FAISS index for fast similarity search
        
        Args:
            embeddings: Document embeddings to index
        """
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        self.faiss_index.add(embeddings)

    def save_processed_data(self, output_path: str = "/Users/mihiranga/Msc/research-code/processed_data") -> None:
        """Save processed letter data for future use"""
        if self.letter_df is None:
            print(f"Processed data saving error {self.letter_df}")
            raise ValueError("No processed data available. Run process_dataset first.")
        
        # Create output directory if it doesn't exist
        Path(output_path).mkdir(parents=True, exist_ok=True)
        print(f"Processed data saving file created {output_path}")
        
        # Save dataframe to CSV for easy viewing (exclude embedding column)
        df_to_save = self.letter_df.copy()
        
        # Rename 'content' to 'sinhala_text' for clarity
        df_to_save.rename(columns={'content': 'sinhala_text'}, inplace=True)
        
        # Drop embedding column and save with UTF-8 encoding to preserve Sinhala characters
        df_to_save.drop(columns=['embedding'], inplace=True)
        
        # Use utf-8-sig encoding to handle BOM for better Excel/CSV reader compatibility
        df_to_save.to_csv(Path(output_path) / "processed_letters.csv", index=False, encoding='utf-8-sig')
        print(f"Processed data saved to csv with UTF-8-SIG encoding for proper Sinhala text display")
        
        # Save JSON version which typically handles Unicode better
        df_to_save.to_json(Path(output_path) / "processed_letters.json", orient='records', force_ascii=False)
        print("Also saved as JSON with proper Unicode support")
        
        # Save dataframe to pickle for programmatic use
        self.letter_df.to_pickle(Path(output_path) / "processed_letters.pkl")
        
        print(f"Saved processed data to {output_path}")

    def save_embeddings(self, output_path: str = "/Users/mihiranga/Msc/research-code/embeddings"):
        """Save letter embeddings and FAISS index for future use
        
        Args:
            output_path: Directory to save embeddings
        """
        if self.letter_df is None:
            raise ValueError("No processed data available. Run process_dataset first.")
        
        # Create output directory
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # Save embeddings directly from DataFrame
        embeddings = np.vstack(self.letter_df['embedding'].values)
        np.save(Path(output_path) / "letter_embeddings.npy", embeddings)
        
        # Save filenames in the same order as embeddings for reference
        filenames = self.letter_df['file_name'].values
        np.save(Path(output_path) / "letter_filenames.npy", filenames)
        
        # Save FAISS index
        if self.faiss_index:
            faiss.write_index(self.faiss_index, str(Path(output_path) / "letter_embeddings.index"))
        
        print(f"Saved {len(self.letter_df)} embeddings and FAISS index to {output_path}")
        
        # Save a mapping of file names to their types for quick reference
        type_mapping = dict(zip(self.letter_df['file_name'], self.letter_df['letter_type']))
        np.save(Path(output_path) / "letter_types_mapping.npy", type_mapping)
    
    def load_embeddings(self, input_path: str = "/Users/mihiranga/Msc/research-code/embeddings"):
        """Load previously saved embeddings and FAISS index
        
        Args:
            input_path: Directory containing saved embeddings and index
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load FAISS index
            index_path = Path(input_path) / "letter_embeddings.index"
            if index_path.exists():
                self.faiss_index = faiss.read_index(str(index_path))
                print(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
            else:
                print(f"FAISS index not found at {index_path}")
                return False
            
            # Load type mapping
            type_mapping_path = Path(input_path) / "letter_types_mapping.npy"
            if type_mapping_path.exists():
                self.letter_types = list(np.load(type_mapping_path, allow_pickle=True).item().values())
                self.letter_types = list(set(self.letter_types))  # Get unique types
                
            return True
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return False

if __name__ == "__main__":
    processor = LetterProcessor()
    processed_df = processor.process_dataset()
    processor.save_processed_data()
    processor.save_embeddings()
    print(f"Processed {len(processed_df)} letters")
    print(f"Letter types found: {processor.letter_types}")
