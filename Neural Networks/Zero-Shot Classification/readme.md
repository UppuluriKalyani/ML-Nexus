
# Zero-Shot Classification Model

This repository contains a Zero-Shot Classification model designed for Natural Language Inference (NLI) tasks. The model classifies sentence pairs into three categories: **contradiction**, **entailment**, and **neutral**.

## Model Overview

The model is based on a transformer architecture and was trained using the `CrossEncoder` approach. It outputs three scores for any given sentence pair, representing the likelihood of each relationship.

### Training Data

The model was trained on two datasets:
- **SNLI (Stanford Natural Language Inference)**: A dataset containing sentence pairs labeled as contradiction, entailment, or neutral.
- **MultiNLI (Multi-Genre Natural Language Inference)**: An extension of SNLI that includes sentence pairs from various genres.

### Performance

- **Accuracy on SNLI-test dataset:** 92.38%
- **Accuracy on MNLI mismatched set:** 90.04%

## Usage

### Using the Model with SentenceTransformers

You can use the pre-trained model as follows:

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
scores = model.predict([
    ('A man is eating pizza', 'A man eats something'),
    ('A black race car starts up in front of a crowd of people.', 'A man is driving down a lonely road.')
])

# Convert scores to labels
label_mapping = ['contradiction', 'entailment', 'neutral']
labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
```

### Using the Model with Transformers Library

The model can also be utilized directly with the Transformers library:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-base')
tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-base')

features = tokenizer(
    ['A man is eating pizza', 'A black race car starts up in front of a crowd of people.'], 
    ['A man eats something', 'A man is driving down a lonely road.'],  
    padding=True, 
    truncation=True, 
    return_tensors="pt"
)

model.eval()
with torch.no_grad():
    scores = model(**features).logits
    label_mapping = ['contradiction', 'entailment', 'neutral']
    labels = [label_mapping[score_max] for score_max in scores.argmax(dim=1)]
    print(labels)
```

### Zero-Shot Classification

This model can also be employed for zero-shot classification:

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model='cross-encoder/nli-deberta-v3-base')

sent = "Apple just announced the newest iPhone X"
candidate_labels = ["technology", "sports", "politics"]
res = classifier(sent, candidate_labels)
print(res)
```
