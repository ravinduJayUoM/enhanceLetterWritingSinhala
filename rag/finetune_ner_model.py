"""
Fine-tune the Sinhala Letter NER model.
This script prepares training data and runs the fine-tuning process.
"""

import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.models.sinhala_ner import create_model
from rag.models.prepare_ner_dataset import main as prepare_dataset

def main():
    """Run the fine-tuning process for the Sinhala Letter NER model."""
    parser = argparse.ArgumentParser(description="Fine-tune the Sinhala Letter NER model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for training")
    parser.add_argument("--model", type=str, default="xlm-roberta-base", help="Base model to use")
    parser.add_argument("--skip-prep", action="store_true", help="Skip dataset preparation")
    args = parser.parse_args()
    
    # Set up paths
    training_data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "training_data"
    )
    
    # Step 1: Prepare the dataset (unless skipped)
    if not args.skip_prep:
        print("=== Step 1: Preparing training dataset ===")
        prepare_dataset()
    else:
        print("=== Skipping dataset preparation ===")
    
    # Step 2: Initialize the model
    print("\n=== Step 2: Initializing Sinhala Letter NER model ===")
    ner_model = create_model(
        model_name=args.model,
        use_spacy=True,
        use_rules=True
    )
    
    # Step 3: Fine-tune the model
    print("\n=== Step 3: Fine-tuning model ===")
    ner_model.fine_tune(
        training_data_path=training_data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    print("\n=== Fine-tuning complete! ===")
    print(f"The fine-tuned model is saved at: {os.path.join(training_data_path, 'best_model')}")

if __name__ == "__main__":
    main()