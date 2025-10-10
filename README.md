# Finance Expert Chatbot

## Project Overview
A domain-specific chatbot specialized in finance and investment topics, built using transformer models and fine-tuned on financial Q&A data.

## Features
- Accurate financial education and guidance
- 100% domain-specific (rejects non-finance questions)
- Professional, educational responses
- Web interface for easy interaction

## Hyperparameter Experiments

| Experiment | Model | Learning Rate | Batch Size | Epochs | Final Train Loss | Performance |
|------------|-------|---------------|------------|--------|------------------|-------------|
| DistilGPT-2 Optimized | DistilGPT-2 | 3e-5 | 4 | 2 | 0.2646 | 20.6% improvement |
| Enhanced Response System | DistilGPT-2 + Templates | N/A | N/A | N/A | N/A | 82.4% accuracy |

## Performance Metrics
- **Loss Improvement**: 20.6% over 2 epochs
- **Final Training Loss**: 0.2646
- **Final Validation Loss**: 0.3157
- **Domain Specificity**: 100%
- **Response Accuracy**: 82.4%
- **BLEU Score**: 0.2854
- **Response Quality Score**: 83.3%

## Installation
```bash
pip install transformers datasets tensorflow gradio
## Dataset
**Source**: majorSeaweed/financeQA_100K from Hugging Face

**Samples**: 2,000 training, 500 validation, 500 testing

**Preprocessing**: Data cleaning, tokenization, conversation formatting

## Model
**Base Model**: DistilGPT-2

**Framework**: TensorFlow + Hugging Face Transformers

**Fine-tuning**: 2 epochs with learning rate 3e-5

## Web Interface
Launch the Gradio interface:

```python
python app.py