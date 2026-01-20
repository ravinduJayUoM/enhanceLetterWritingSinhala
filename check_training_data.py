import json

# Check raw training data
with open(r'c:\MSC\code\enhanceLetterWritingSinhala\rag\models\training_data\raw\train_samples.json', encoding='utf-8') as f:
    train_data = json.load(f)

with open(r'c:\MSC\code\enhanceLetterWritingSinhala\rag\models\training_data\raw\val_samples.json', encoding='utf-8') as f:
    val_data = json.load(f)

print("=== RAW TRAINING DATA STATISTICS ===")
print(f"Total train samples: {len(train_data)}")
print(f"Total validation samples: {len(val_data)}")

annotated_train = [d for d in train_data if d.get("entities") and any(d["entities"].values())]
annotated_val = [d for d in val_data if d.get("entities") and any(d["entities"].values())]

print(f"\nAnnotated train samples: {len(annotated_train)}")
print(f"Annotated validation samples: {len(annotated_val)}")
print(f"Unannotated train samples: {len(train_data) - len(annotated_train)}")
print(f"Unannotated validation samples: {len(val_data) - len(annotated_val)}")

# Check entity distribution
entity_counts = {}
for sample in annotated_train:
    for entity_type in sample["entities"].keys():
        entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

print("\n=== ENTITY TYPE DISTRIBUTION (Train) ===")
for entity_type, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{entity_type}: {count}")

# Check transformer format
try:
    with open(r'c:\MSC\code\enhanceLetterWritingSinhala\rag\models\training_data\transformer\train.json', encoding='utf-8') as f:
        transformer_train = json.load(f)
    
    with open(r'c:\MSC\code\enhanceLetterWritingSinhala\rag\models\training_data\transformer\val.json', encoding='utf-8') as f:
        transformer_val = json.load(f)
    
    print(f"\n=== TRANSFORMER FORMAT DATA ===")
    print(f"Transformer train samples: {len(transformer_train)}")
    print(f"Transformer validation samples: {len(transformer_val)}")
except Exception as e:
    print(f"\nTransformer format data not found or error: {e}")

print("\n=== SAMPLE ANNOTATED EXAMPLE ===")
if annotated_train:
    sample = annotated_train[0]
    print(f"Text preview: {sample['text'][:200]}...")
    print(f"\nEntities:")
    for entity_type, entity_data in sample['entities'].items():
        if entity_data:
            print(f"  - {entity_type}: {entity_data.get('text', 'N/A')[:100]}")
