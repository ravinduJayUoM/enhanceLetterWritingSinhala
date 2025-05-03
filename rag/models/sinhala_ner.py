"""
Custom Named Entity Recognition (NER) model for Sinhala letter information extraction.
This model extracts structured information from Sinhala letter requests.
"""

import re
import os
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import json

# Default model settings
DEFAULT_MODEL_NAME = "xlm-roberta-base"  # Multilingual model that supports Sinhala
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SinhalaLetterNER:
    """Custom NER model for Sinhala letter information extraction"""
    
    def __init__(
        self, 
        model_name: str = DEFAULT_MODEL_NAME,
        use_spacy: bool = True,
        use_rules: bool = True
    ):
        """
        Initialize the Sinhala Letter NER model.
        
        Args:
            model_name: The name of the pretrained transformer model to use
            use_spacy: Whether to use spaCy for additional NER
            use_rules: Whether to use rule-based extraction as a fallback
        """
        self.model_name = model_name
        self.use_spacy = use_spacy
        self.use_rules = use_rules
        self.nlp = None
        self.tokenizer = None
        self.model = None
        
        # Load transformer model
        if model_name != "simple-rule-based":
            try:
                from transformers import AutoTokenizer, AutoModel
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.model.to(DEVICE)
                print(f"Loaded transformer model: {model_name}")
            except Exception as e:
                print(f"Error loading transformer model: {str(e)}")
                print("Falling back to rule-based extraction only")
                self.model = None
                self.tokenizer = None
                self.use_rules = True
        
        # Load spaCy model if enabled
        if use_spacy:
            try:
                # Load spaCy model for Sinhala if available, otherwise use multilingual model
                import spacy
                try:
                    self.nlp = spacy.load("xx_ent_wiki_sm")  # Multilingual model
                    print("Loaded spaCy multilingual model for additional NER")
                except:
                    print("SpaCy multilingual model not found. Try running: python -m spacy download xx_ent_wiki_sm")
                    try:
                        self.nlp = spacy.blank("si")  # Create blank Sinhala model
                        print("Created blank Sinhala model")
                    except:
                        print("Could not create blank Sinhala model")
                        self.nlp = None
                        self.use_spacy = False
            except ImportError as e:
                print(f"Error importing spaCy: {str(e)}")
                self.nlp = None
                self.use_spacy = False

        # Entity types to extract
        self.entity_types = [
            "letter_type", "recipient", "sender", "subject", 
            "purpose", "details", "qualifications", "contact_details",
            "incident_date", "requested_action", "requested_items", "timeline"
        ]
        
        # Compile regex patterns for rule-based extraction
        self.patterns = self._compile_patterns()
        
        # Example phrases for each entity type (for similarity matching)
        self.example_phrases = self._load_example_phrases()
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for rule-based extraction"""
        patterns = {
            "letter_type": re.compile(r'(අයදුම්පත|ඉල්ලීම|පැමිණිල්ල|දැනුම්දීම|අභියාචනය)', re.IGNORECASE),
            "recipient": re.compile(r'(වෙත|බලා|අමතා|යොමු කරන)', re.IGNORECASE),
            "sender": re.compile(r'(මගින්|විසින්|වෙතින්|අත්සන)', re.IGNORECASE),
            "subject": re.compile(r'(මාතෘකාව|විෂය|හේතුව)[:]\s*(.*)', re.IGNORECASE),
            "purpose": re.compile(r'(අරමුණ|හේතුව|කාරණය|ඉල්ලා සිටිමි|කාරුණිකව ඉල්ලමි)[:]*\s*(.*)', re.IGNORECASE),
            "contact_details": re.compile(r'(දුරකථන අංකය|විද්‍යුත් තැපෑල|ලිපිනය)[:]\s*(.*)', re.IGNORECASE),
            "incident_date": re.compile(r'(දිනය|කාලය)[:]\s*(.*)', re.IGNORECASE),
        }
        return patterns
    
    def _load_example_phrases(self) -> Dict[str, List[str]]:
        """Load example phrases for each entity type for similarity matching"""
        # These phrases help the model identify similar text in the input
        return {
            "letter_type": [
                "මෙය රැකියා අයදුම්පතකි", "මෙම ලිපිය ඉල්ලීමක්", "මෙය පැමිණිල්ලකි", 
                "මෙම දැනුම්දීම", "අභියාචනයක්"
            ],
            "recipient": [
                "පාසල් විදුහල්පති වෙත", "සභාපතිතුමා වෙත", "අධ්‍යක්ෂක ජනරාල් වෙත",
                "අමාත්‍යාංශ ලේකම් වෙත", "කළමනාකරු වෙත"
            ],
            "sender": [
                "මා විසින්", "මගේ නම", "අත්සන් කළේ", "ඉදිරිපත් කරන්නේ", 
                "මම වන", "මගේ ලිපිනය"
            ],
            "subject": [
                "විෂය:", "මාතෘකාව:", "කරුණ:", "මෙම ලිපියේ අරමුණ",
                "ඉල්ලීම සම්බන්ධයෙන්"
            ],
            "purpose": [
                "අරමුණ වන්නේ", "කරුණ වන්නේ", "හේතුව වන්නේ", 
                "මෙම ඉල්ලීමේ පරමාර්ථය", "ලිපියේ අරමුණ"
            ],
        }
    
    def _rule_based_extraction(self, text: str) -> Dict[str, Any]:
        """Extract information using rule-based patterns"""
        results = {}
        
        # Apply patterns
        for entity_type, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                if entity_type in ["subject", "purpose", "contact_details", "incident_date"]:
                    # For patterns that capture the full value
                    results[entity_type] = matches[0][1] if isinstance(matches[0], tuple) else matches[0]
                else:
                    # For patterns that just identify presence
                    # Extract the sentence containing the match
                    for match in matches:
                        match_str = match[0] if isinstance(match, tuple) else match
                        idx = text.find(match_str)
                        if idx >= 0:
                            # Extract the sentence containing this match
                            sentence_start = text.rfind('.', 0, idx) + 1
                            sentence_end = text.find('.', idx)
                            if sentence_end < 0:
                                sentence_end = len(text)
                            
                            sentence = text[sentence_start:sentence_end].strip()
                            results[entity_type] = sentence
                            break
        
        # Look for first line recipient
        lines = text.split('\n')
        if lines and not results.get("recipient") and len(lines[0]) < 100:
            results["recipient"] = lines[0].strip()
        
        # Look for last lines for sender
        if not results.get("sender") and len(lines) > 3:
            for line in lines[-5:]:
                if len(line.strip()) > 5 and len(line.strip()) < 100:
                    # Look for names that might indicate a sender
                    if any(word in line.lower() for word in ["අත්සන", "විසින්", "ඉදිරිපත් කරන්නේ", "මහතා", "මහත්මිය"]):
                        results["sender"] = line.strip()
                        break
        
        # Extract letter type based on keywords
        if "letter_type" not in results:
            if any(word in text.lower() for word in ["අයදුම්පත", "සුදුසුකම්", "රැකියාව"]):
                results["letter_type"] = "application"
            elif any(word in text.lower() for word in ["ඉල්ලීම", "ඉල්ලා සිටිමි"]):
                results["letter_type"] = "request"
            elif any(word in text.lower() for word in ["පැමිණිල්ල", "දුක්ගැනවිල්ල"]):
                results["letter_type"] = "complaint"
            elif any(word in text.lower() for word in ["දැනුම්දීම", "නිවේදනය"]):
                results["letter_type"] = "notice"
            else:
                results["letter_type"] = "general"
        
        # Look for contact details (phone, email)
        if not results.get("contact_details"):
            # Look for email pattern
            email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
            email_matches = email_pattern.findall(text)
            
            # Look for phone pattern
            phone_pattern = re.compile(r'\b\d{10}\b|\b\d{3}[-\s]?\d{7}\b')
            phone_matches = phone_pattern.findall(text)
            
            if email_matches or phone_matches:
                contact_details = []
                if email_matches:
                    contact_details.append(f"Email: {email_matches[0]}")
                if phone_matches:
                    contact_details.append(f"Phone: {phone_matches[0]}")
                
                results["contact_details"] = ", ".join(contact_details)
        
        return results
    
    def _spacy_based_extraction(self, text: str) -> Dict[str, Any]:
        """Extract information using spaCy NER"""
        if not self.nlp:
            return {}
        
        results = {}
        doc = self.nlp(text)
        
        # Extract entities recognized by spaCy
        for ent in doc.ents:
            if ent.label_ == "PER" or ent.label_ == "PERSON":
                # Could be sender or recipient
                if "sender" not in results:
                    results["sender"] = ent.text
                elif "recipient" not in results:
                    results["recipient"] = ent.text
            
            elif ent.label_ == "ORG" or ent.label_ == "ORGANIZATION":
                # Likely a recipient if organization
                if "recipient" not in results:
                    results["recipient"] = ent.text
            
            elif ent.label_ == "DATE":
                if "incident_date" not in results:
                    results["incident_date"] = ent.text
        
        return results
    
    def _transformer_based_extraction(self, text: str) -> Dict[str, Any]:
        """Extract information using transformer model"""
        if not self.model or not self.tokenizer:
            return {}
        
        results = {}
        
        # Break text into sentences
        sentences = re.split(r'[.!?።]\s+', text)
        
        # Create embeddings for each sentence
        sentence_embeddings = []
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Tokenize and get embeddings
            inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Use CLS token as sentence embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            sentence_embeddings.append((sentence, embedding[0]))
        
        # Create embeddings for example phrases
        phrase_embeddings = {}
        for entity_type, phrases in self.example_phrases.items():
            phrase_embeddings[entity_type] = []
            for phrase in phrases:
                inputs = self.tokenizer(phrase, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                phrase_embeddings[entity_type].append(embedding[0])
        
        # Compare sentence embeddings with example phrase embeddings
        for sentence, sent_embedding in sentence_embeddings:
            best_match = (None, -1)  # (entity_type, similarity)
            
            for entity_type, embeddings in phrase_embeddings.items():
                for emb in embeddings:
                    # Calculate cosine similarity
                    similarity = np.dot(sent_embedding, emb) / (np.linalg.norm(sent_embedding) * np.linalg.norm(emb))
                    if similarity > best_match[1] and similarity > 0.6:  # Threshold
                        best_match = (entity_type, similarity)
            
            if best_match[0] and best_match[0] not in results:
                results[best_match[0]] = sentence
        
        return results
    
    def extract_info(self, text: str) -> Dict[str, Any]:
        """
        Extract structured information from Sinhala text.
        
        Args:
            text: The Sinhala text to extract information from
            
        Returns:
            Dictionary containing extracted entities
        """
        results = {}
        
        # Use multiple extraction methods and combine results
        if self.use_rules:
            rule_results = self._rule_based_extraction(text)
            results.update(rule_results)
        
        if self.use_spacy and self.nlp:
            spacy_results = self._spacy_based_extraction(text)
            for k, v in spacy_results.items():
                if k not in results:
                    results[k] = v
        
        if self.model and self.tokenizer:
            transformer_results = self._transformer_based_extraction(text)
            for k, v in transformer_results.items():
                if k not in results:
                    results[k] = v
        
        # Extract other fields like details by removing identified sections
        identified_text = ' '.join(results.values())
        remaining_text = text
        for value in results.values():
            remaining_text = remaining_text.replace(value, '')
        
        if len(remaining_text.strip()) > 0 and "details" not in results:
            results["details"] = remaining_text.strip()
        
        # Set default value for missing fields
        for field in self.entity_types:
            if field not in results:
                results[field] = ""
        
        return results
    
    def fine_tune(self, training_data_path: str, epochs: int = 3, batch_size: int = 8, learning_rate: float = 2e-5):
        """
        Fine-tune the model with labeled Sinhala letter data.
        
        Args:
            training_data_path: Path to the training data directory with transformer format files
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for training
        """
        # Check if transformer models are available
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            import torch.optim as optim
            from torch.utils.data import Dataset, DataLoader
            from tqdm import tqdm
        except ImportError:
            print("Transformers or PyTorch not available. Run 'pip install transformers torch tqdm'")
            return
        
        # Check if we have a transformer model to fine-tune
        if not self.model or not self.tokenizer:
            print("No transformer model available for fine-tuning. Initializing a new one.")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForTokenClassification.from_pretrained(
                    self.model_name,
                    num_labels=len(self.entity_types) * 2 + 1  # BIO tagging: O + B/I for each entity
                )
                self.model.to(DEVICE)
            except Exception as e:
                print(f"Failed to initialize model for fine-tuning: {str(e)}")
                return
        
        print(f"Starting fine-tuning with data from {training_data_path}")
        
        # Define a custom dataset for NER
        class NERDataset(Dataset):
            def __init__(self, data_file, tokenizer, max_length=128):
                with open(data_file, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                self.tokenizer = tokenizer
                self.max_length = max_length
                
                # Create label mapping
                self.label_map = {"O": 0}  # Outside any entity
                entity_types = [
                    "letter_type", "recipient", "sender", "subject", 
                    "purpose", "details", "qualifications", "contact_details",
                    "incident_date", "requested_action", "requested_items", "timeline"
                ]
                
                idx = 1
                for entity in entity_types:
                    self.label_map[f"B-{entity}"] = idx
                    idx += 1
                    self.label_map[f"I-{entity}"] = idx
                    idx += 1
                
                self.num_labels = len(self.label_map)
                
            def __len__(self):
                return len(self.data)
                
            def __getitem__(self, idx):
                item = self.data[idx]
                text = " ".join(item["tokens"]) if "tokens" in item else item.get("text", "")
                labels = item.get("labels", ["O"] * len(text.split()))
                
                # Tokenize input
                encoding = self.tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                # Remove batch dimension
                input_ids = encoding["input_ids"].squeeze()
                attention_mask = encoding["attention_mask"].squeeze()
                
                # Convert string labels to ids
                label_ids = torch.tensor([self.label_map.get(label, 0) for label in labels])
                
                # Pad or truncate label_ids to match input_ids
                if len(label_ids) < len(input_ids):
                    # Pad labels
                    padding = torch.zeros(len(input_ids) - len(label_ids), dtype=torch.long)
                    label_ids = torch.cat([label_ids, padding])
                else:
                    # Truncate labels
                    label_ids = label_ids[:len(input_ids)]
                
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": label_ids
                }
        
        # Check if training data exists
        train_file = os.path.join(training_data_path, "transformer", "train.json")
        val_file = os.path.join(training_data_path, "transformer", "val.json")
        
        if not os.path.exists(train_file) or not os.path.exists(val_file):
            print(f"Training data files not found at {training_data_path}")
            # Try to prepare the dataset
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            try:
                from models.prepare_ner_dataset import main as prepare_dataset
                print("Running dataset preparation...")
                prepare_dataset()
            except ImportError:
                print("Could not import dataset preparation module")
                return
            
            # Check again after preparation
            if not os.path.exists(train_file) or not os.path.exists(val_file):
                print(f"Failed to create training data")
                return
        
        try:
            # Create datasets and dataloaders
            train_dataset = NERDataset(train_file, self.tokenizer)
            val_dataset = NERDataset(val_file, self.tokenizer)
            
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Initialize model for token classification if needed
            if not isinstance(self.model, AutoModelForTokenClassification):
                print("Converting model to token classification model")
                self.model = AutoModelForTokenClassification.from_pretrained(
                    self.model_name,
                    num_labels=train_dataset.num_labels
                )
                self.model.to(DEVICE)
            
            # Set up optimizer
            optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
            
            # Training loop
            print(f"Starting training for {epochs} epochs...")
            best_f1 = 0
            
            for epoch in range(epochs):
                # Training phase
                self.model.train()
                train_loss = 0
                train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Training]")
                
                for batch in train_pbar:
                    input_ids = batch["input_ids"].to(DEVICE)
                    attention_mask = batch["attention_mask"].to(DEVICE)
                    labels = batch["labels"].to(DEVICE)
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    train_loss += loss.item()
                    
                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    train_pbar.set_postfix({"loss": loss.item()})
                
                avg_train_loss = train_loss / len(train_dataloader)
                
                # Validation phase
                self.model.eval()
                val_loss = 0
                all_preds = []
                all_labels = []
                
                with torch.no_grad():
                    val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Validation]")
                    for batch in val_pbar:
                        input_ids = batch["input_ids"].to(DEVICE)
                        attention_mask = batch["attention_mask"].to(DEVICE)
                        labels = batch["labels"].to(DEVICE)
                        
                        # Forward pass
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        
                        loss = outputs.loss
                        val_loss += loss.item()
                        
                        # Get predictions
                        logits = outputs.logits
                        preds = torch.argmax(logits, dim=2)
                        
                        # Only consider predictions where attention mask is 1
                        active_preds = preds[attention_mask == 1]
                        active_labels = labels[attention_mask == 1]
                        
                        all_preds.extend(active_preds.detach().cpu().numpy())
                        all_labels.extend(active_labels.detach().cpu().numpy())
                        
                        val_pbar.set_postfix({"loss": loss.item()})
                
                avg_val_loss = val_loss / len(val_dataloader)
                
                # Try to calculate metrics
                try:
                    from sklearn.metrics import classification_report
                    # Calculate metrics
                    label_names = list(train_dataset.label_map.keys())
                    report = classification_report(
                        all_labels, 
                        all_preds, 
                        output_dict=True,
                        labels=list(train_dataset.label_map.values()),
                        target_names=label_names
                    )
                    
                    # Calculate macro F1 score
                    macro_f1 = report["macro avg"]["f1-score"]
                    
                    print(f"Epoch {epoch+1}/{epochs}:")
                    print(f"  Train Loss: {avg_train_loss:.4f}")
                    print(f"  Val Loss: {avg_val_loss:.4f}")
                    print(f"  Macro F1: {macro_f1:.4f}")
                    
                    # Save the best model
                    if macro_f1 > best_f1:
                        best_f1 = macro_f1
                        model_path = os.path.join(os.path.dirname(training_data_path), "best_model")
                        os.makedirs(model_path, exist_ok=True)
                        
                        self.model.save_pretrained(model_path)
                        self.tokenizer.save_pretrained(model_path)
                        print(f"  Saved best model with F1: {best_f1:.4f}")
                except ImportError:
                    print(f"Epoch {epoch+1}/{epochs}:")
                    print(f"  Train Loss: {avg_train_loss:.4f}")
                    print(f"  Val Loss: {avg_val_loss:.4f}")
                    print("  sklearn not available for metrics calculation")
            
            print(f"Fine-tuning completed. Best F1: {best_f1:.4f}")
            
        except Exception as e:
            import traceback
            print(f"Error during fine-tuning: {str(e)}")
            traceback.print_exc()
            return

# Function to create and return a pre-configured model
def create_model(
    model_name: str = DEFAULT_MODEL_NAME,
    use_spacy: bool = True,
    use_rules: bool = True
) -> SinhalaLetterNER:
    """
    Create a pre-configured Sinhala Letter NER model.
    
    Args:
        model_name: Name of transformer model to use
        use_spacy: Whether to use spaCy NER
        use_rules: Whether to use rule-based extraction
        
    Returns:
        Configured SinhalaLetterNER model
    """
    return SinhalaLetterNER(
        model_name=model_name,
        use_spacy=use_spacy,
        use_rules=use_rules
    )