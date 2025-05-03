"""
Prepare a fine-tuning dataset for the Sinhala Letter NER model.
This script processes existing Sinhala letters to create labeled training data.
"""

import os
import re
import json
import glob
import pandas as pd
import random
from typing import List, Dict, Any, Tuple

# Path to the letter dataset
LETTERS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "training_data")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Entity types we want to extract
ENTITY_TYPES = [
    "letter_type", "recipient", "sender", "subject", 
    "purpose", "details", "qualifications", "contact_details",
    "incident_date", "requested_action", "requested_items", "timeline"
]

# Keywords to identify entity types in Sinhala text
ENTITY_KEYWORDS = {
    "letter_type": [
        "අයදුම්පත", "ඉල්ලීම", "පැමිණිල්ල", "දැනුම්දීම", "අභියාචනය",
        "අයදුම්", "ඉල්ලුම්", "පිළිගැනීම"
    ],
    "recipient": [
        "වෙත", "බලා", "අමතා", "යොමු කරන", "හට",
        "මහතා වෙත", "මහත්මිය වෙත", "දෙපාර්තමේන්තුව", "කාර්යාලය"
    ],
    "sender": [
        "මගින්", "විසින්", "වෙතින්", "අත්සන", "ඉදිරිපත් කරන්නේ",
        "ලියන", "යවන", "ඉදිරිපත් කරන"
    ],
    "subject": [
        "මාතෘකාව", "විෂය", "කරුණ", "සම්බන්ධව", "සම්බන්ධයෙන්",
        "පිළිබඳව", "ඉල්ලීම", "සඳහා"
    ],
    "purpose": [
        "අරමුණ", "හේතුව", "කාරණය", "බලාපොරොත්තුව", "පරමාර්ථය",
        "මෙම ලිපියෙන්", "ඉල්ලා සිටිමි", "දන්වා සිටිමි"
    ],
    "contact_details": [
        "දුරකථන", "විද්‍යුත් තැපෑල", "ලිපිනය", "සම්බන්ධ කර ගත හැකි",
        "ජංගම දුරකථන", "ස්ථාවර දුරකථන", "ඊමේල්"
    ],
    "incident_date": [
        "දිනය", "කාලය", "වකවානුව", "සිද්ධිය", "අවස්ථාව",
        "සිදුවූ දිනය", "දා", "සිට", "දක්වා"
    ],
}

# Patterns to identify letter sections (updated with new info)
SECTION_PATTERNS = {
    # First line is often the heading
    "heading": r"^.*$",
    # Greeting is often one of these phrases
    "greeting": r"(මහත්මයාණෙනි|මහත්මියනි|මහතාණෙනි|මහත්මයාණෙනි|මහත්මියනි|මහතාණෙනි)",
    # Recipient's address is just above the date (date is right below recipient address)
    # This can take up to several lines, so we look for lines above a date pattern
    "recipient_address": r"((?:.*\n)+?)(?=\d{4}[/-]\d{1,2}[/-]\d{1,2}|දිනය|කාලය)",  # lines above a date
    # Date pattern (for finding recipient/sender address boundaries)
    "date": r"(\d{4}[/-]\d{1,2}[/-]\d{1,2}|දිනය[:\-]?\s*\d{4}[/-]\d{1,2}[/-]\d{1,2})",
    # Sender's address is right below the date, can take up to several lines, ends before subject or body
    "sender_address": r"(?<=\d{4}[/-]\d{1,2}[/-]\d{1,2}|දිනය[:\-]?.*)\n((?:.*\n)+?)(?=(මාතෘකාව|විෂය|කරුණ|^.{0,30}$))",
    # Subject is right below sender's address, sometimes doesn't contain the given words
    # Try to match a short line after sender address or date
    "subject_section": r"(මාතෘකාව|විෂය|කරුණ)[:]?\s*(.*)|^(?P<subject>.{5,50})$",  # fallback: any short line
    # Body starts after subject
    "body_start": r"^[^\n]+[\n]",  # First paragraph after greeting/subject
    # Signature often has "මෙයට විශ්වාසී" or "මෙයට"
    "signature": r".*(මෙයට විශ්වාසී|මෙයට|අත්සන|ස්තූතියි).*"
}

def read_letter_files(directory: str) -> List[str]:
    """Read all text files from the specified directory."""
    letter_files = glob.glob(os.path.join(directory, "*.txt"))
    letters = []
    
    for file_path in letter_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                if content and len(content.strip()) > 50:  # Ensure it's a substantial letter
                    letters.append(content)
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
    
    print(f"Read {len(letters)} letters from {directory}")
    return letters

def extract_entities_from_letter(letter: str) -> Dict[str, str]:
    """
    Extract entities from a letter using rule-based patterns.
    This is a simple first pass that will be improved through fine-tuning.
    """
    entities = {entity_type: "" for entity_type in ENTITY_TYPES}
    
    # Split letter into lines for processing
    lines = letter.split('\n')
    
    # Extract letter type from the first few lines
    first_three_lines = ' '.join(lines[:3]).lower()
    for entity_type, keywords in ENTITY_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in first_three_lines:
                # Find the line containing this keyword
                for line in lines[:5]:
                    if keyword.lower() in line.lower():
                        entities[entity_type] = line.strip()
                        break
                break
    
    # Extract recipient (usually in the first few lines)
    for i, line in enumerate(lines[:5]):
        if "වෙත" in line or "බලා" in line or "අමතා" in line:
            entities["recipient"] = line.strip()
            break
    
    # Extract subject (often has a label)
    for line in lines:
        subject_match = re.search(r"(මාතෘකාව|විෂය|කරුණ)[:]?\s*(.*)", line, re.IGNORECASE)
        if subject_match:
            entities["subject"] = subject_match.group(2).strip()
            break
    
    # Extract sender (usually at the end)
    for line in lines[-5:]:
        if any(keyword in line.lower() for keyword in ["අත්සන", "විසින්", "ඉදිරිපත් කරන්නේ"]):
            entities["sender"] = line.strip()
            break
    
    # Extract contact details
    for line in lines:
        if any(keyword in line.lower() for keyword in ["දුරකථන", "විද්‍යුත්", "ලිපිනය", "ඊමේල්"]):
            entities["contact_details"] = line.strip()
            break
    
    # Try to determine purpose from the body
    body_text = ' '.join(lines[5:-5])  # Rough estimate of the body
    for keyword in ENTITY_KEYWORDS["purpose"]:
        purpose_pattern = re.compile(f"([^.]*{keyword}[^.]*)", re.IGNORECASE)
        purpose_match = purpose_pattern.search(body_text)
        if purpose_match:
            entities["purpose"] = purpose_match.group(1).strip()
            break
    
    # Extract details as the longest section without other entities
    # This is a simplified approach - in real training we'd want proper spans
    if body_text:
        entities["details"] = body_text[:200].strip()  # Take first 200 chars as sample
    
    return entities

def create_training_samples(letters: List[str], sample_count: int = 100) -> List[Dict[str, Any]]:
    """
    Create training samples from letters.
    Each sample contains the text and entity spans.
    """
    samples = []
    
    # Randomly sample letters if we have more than we need
    if len(letters) > sample_count:
        selected_letters = random.sample(letters, sample_count)
    else:
        selected_letters = letters
    
    for letter in selected_letters:
        # Extract entities using rule-based approach (for initial labels)
        entity_texts = extract_entities_from_letter(letter)
        
        # Create a training sample with spans
        sample = {
            "text": letter,
            "entities": {}
        }
        
        # Find spans for each entity in the text
        for entity_type, entity_text in entity_texts.items():
            if entity_text:
                # Use a simple span detection - in production we would use more sophisticated methods
                start_idx = letter.find(entity_text)
                if start_idx >= 0:
                    end_idx = start_idx + len(entity_text)
                    sample["entities"][entity_type] = {
                        "text": entity_text,
                        "start": start_idx,
                        "end": end_idx
                    }
        
        samples.append(sample)
    
    return samples

def save_dataset(samples: List[Dict[str, Any]], file_path: str):
    """Save the dataset to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(samples)} training samples to {file_path}")

def create_finetuning_splits(samples: List[Dict[str, Any]], train_ratio: float = 0.8):
    """Split the dataset into training and validation sets."""
    random.shuffle(samples)
    split_idx = int(len(samples) * train_ratio)
    
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    return train_samples, val_samples

def convert_to_transformer_format(samples: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert our samples to a format suitable for fine-tuning transformer models.
    Returns a DataFrame with text and labels columns.
    """
    data = []
    
    for sample in samples:
        text = sample["text"]
        
        # Create IOB tags (Inside, Outside, Beginning)
        # 'O' for tokens that are not part of any entity
        # 'B-{entity_type}' for the first token of an entity
        # 'I-{entity_type}' for subsequent tokens of an entity
        
        # Here we'd normally use a proper tokenizer, but for this example
        # we'll use a simplified approach with character spans
        
        tokens = []
        labels = []
        
        # Very simplified tokenization for example purposes
        # In real implementation, you'd use a proper Sinhala tokenizer
        words = re.findall(r'\S+', text)
        current_position = 0
        
        for word in words:
            word_start = text.find(word, current_position)
            if word_start == -1:
                continue
                
            word_end = word_start + len(word)
            current_position = word_end
            
            # Check if this word is part of an entity
            entity_label = "O"  # Default is Outside any entity
            
            for entity_type, entity_info in sample["entities"].items():
                entity_start = entity_info["start"]
                entity_end = entity_info["end"]
                
                # If word overlaps with entity
                if word_start >= entity_start and word_start < entity_end:
                    # Beginning of entity
                    if word_start == entity_start:
                        entity_label = f"B-{entity_type}"
                    else:
                        entity_label = f"I-{entity_type}"
                    break
            
            tokens.append(word)
            labels.append(entity_label)
        
        data.append({
            "tokens": tokens,
            "labels": labels,
            "text": text
        })
    
    return pd.DataFrame(data)

def main():
    """Main function to create the fine-tuning dataset."""
    print("Reading letter files...")
    letters = read_letter_files(LETTERS_DIR)
    
    if not letters:
        print("No letter files found!")
        return
    
    print(f"Creating training samples from {len(letters)} letters...")
    samples = create_training_samples(letters)
    
    # Split into training and validation sets
    train_samples, val_samples = create_finetuning_splits(samples)
    
    # Save the raw samples
    os.makedirs(os.path.join(OUTPUT_DIR, "raw"), exist_ok=True)
    save_dataset(train_samples, os.path.join(OUTPUT_DIR, "raw", "train_samples.json"))
    save_dataset(val_samples, os.path.join(OUTPUT_DIR, "raw", "val_samples.json"))
    
    # Convert to formats for different model types
    os.makedirs(os.path.join(OUTPUT_DIR, "transformer"), exist_ok=True)
    
    # Convert to transformer format
    train_df = convert_to_transformer_format(train_samples)
    val_df = convert_to_transformer_format(val_samples)
    
    # Save transformer format data
    train_df.to_json(os.path.join(OUTPUT_DIR, "transformer", "train.json"), orient="records", force_ascii=False, indent=2)
    val_df.to_json(os.path.join(OUTPUT_DIR, "transformer", "val.json"), orient="records", force_ascii=False, indent=2)
    
    print(f"Successfully created training data in {OUTPUT_DIR}")
    print(f"- Raw samples: {len(samples)}")
    print(f"- Training samples: {len(train_samples)}")
    print(f"- Validation samples: {len(val_samples)}")
    
    # Print entity distribution
    entity_counts = {entity_type: 0 for entity_type in ENTITY_TYPES}
    for sample in samples:
        for entity_type in sample["entities"]:
            entity_counts[entity_type] += 1
    
    print("\nEntity distribution in dataset:")
    for entity_type, count in entity_counts.items():
        print(f"- {entity_type}: {count} instances ({count/len(samples)*100:.1f}%)")

if __name__ == "__main__":
    main()