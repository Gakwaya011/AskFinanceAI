# 💬 Finance Chatbot with Fine-Tuned DistilGPT-2

📋 **Project Overview**

A domain-specific finance chatbot built using a fine-tuned DistilGPT-2 transformer model. This project demonstrates advanced NLP techniques including custom TensorFlow training loops, hybrid AI architecture, and comprehensive experiment tracking to create an intelligent financial advisory assistant.

**Model Repository:** [brillant024/finance-chatbot-model](https://huggingface.co/brillant024/finance-chatbot-model)  
**Experiment Tracking:** [W&B Dashboard](https://wandb.ai/c-gakwaya-african-leadership-academy/finance-chatbot-optimized)

---

## 🎯 Key Features

- 🤖 **Fine-Tuned AI Model**: Custom-trained DistilGPT-2 (82M parameters) on financial Q&A dataset
- 💡 **Hybrid Architecture**: Combines rule-based reliability with AI-powered generative responses
- 🎯 **Domain Specialization**: 100% focused on finance, investing, and personal finance topics
- 📊 **Experiment Tracking**: Real-time monitoring with Weights & Biases
- 🚀 **Production Ready**: Deployed model on Hugging Face Hub

---

## 🏗️ System Architecture

\```
┌─────────────────────────────────────────────────┐
│            FINANCE CHATBOT SYSTEM               │
├─────────────────────────────────────────────────┤
│  User Input → Domain Filter → Response Router   │
│                    ↓                            │
│         ┌──────────┴──────────┐                │
│         ↓                     ↓                │
│  Fine-Tuned DistilGPT-2   Rule-Based System    │
│         ↓                     ↓                │
│         └──────────┬──────────┘                │
│                    ↓                            │
│           Quality Evaluation                    │
│                    ↓                            │
│            Response Delivery                    │
└─────────────────────────────────────────────────┘
\```

**Components:**
- **Model**: Fine-tuned DistilGPT-2 (82M parameters)
- **Framework**: TensorFlow 2.19 + Hugging Face Transformers
- **Training**: Custom manual training loop for full control
- **Tracking**: Weights & Biases for experiment monitoring
- **Fallback**: Enhanced rule-based system with 15+ predefined responses

---

## 📈 Model Training & Performance

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | DistilGPT-2 |
| Dataset | financeQA_100K |
| Training Samples | 2,000 |
| Validation Samples | 500 |
| Test Samples | 500 |
| Epochs | 4 |
| Learning Rate | 5e-5 |
| Batch Size | 8 |
| Max Sequence Length | 256 tokens |
| Optimizer | Adam |

### Performance