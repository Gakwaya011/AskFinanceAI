import gradio as gr
from transformers import pipeline
import re
import csv
from datetime import datetime

# -------------------------------
# 1️⃣ Load fine-tuned model
# -------------------------------
model_name = "brillant024/finance-chatbot-model"

try:
    chatbot = pipeline("text-generation", model=model_name)
except Exception as e:
    chatbot = None
    print(f"⚠️ Model could not be loaded: {e}")

# -------------------------------
# 2️⃣ Rule-based finance knowledge base
# -------------------------------
predefined_responses = {
    "what is investing": "Investing means putting money into assets like stocks, bonds, or real estate with the goal of growing wealth over time.",
    "explain stocks": "Stocks represent ownership shares in a company. When you buy stocks, you own part of that company.",
    "how do bonds work": "Bonds are loans you make to companies or governments, earning fixed interest over time.",
    "what is compound interest": "Compound interest means earning interest on both your original amount and the interest already earned — this accelerates growth.",
    "how do i start investing": "Start by learning the basics, setting goals, building an emergency fund, and diversifying your investments.",
    "what are mutual funds": "Mutual funds pool money from many investors to buy a diversified mix of stocks, bonds, or other assets.",
    "difference between stocks and bonds": "Stocks give ownership in a company; bonds are loans to a company or government. Stocks have higher risk and return potential.",
    "what is diversification": "Diversification means spreading investments across assets to reduce risk — don’t put all your eggs in one basket.",
    "what is a portfolio": "A portfolio is your collection of investments — like stocks, bonds, or real estate — tailored to your financial goals.",
    "inflation and investing": "Inflation reduces money’s purchasing power. Investing helps your money grow faster than inflation erodes it.",
    "what are dividends": "Dividends are payments to shareholders from company profits — not all companies offer them.",
    "what is risk in investing": "Risk means the chance of losing money on an investment. Higher risk often brings higher potential returns.",
    "what is etf": "ETFs are funds that trade like stocks and usually track an index. They offer low fees and diversification.",
    "how do retirement accounts work": "Retirement accounts (like 401k or IRA) offer tax benefits for saving long-term.",
    "what is asset allocation": "Asset allocation means dividing your investments among asset types (stocks, bonds, etc.) to balance risk and reward."
}

# -------------------------------
# 3️⃣ Finance keywords for topic detection
# -------------------------------
finance_keywords = [
    'invest', 'stock', 'bond', 'interest', 'money', 'finance', 'financial',
    'saving', 'retirement', 'loan', 'credit', 'debt', 'inflation', 'tax',
    'portfolio', 'fund', 'asset', 'dividend', 'risk', 'etf', 'ira', '401k'
]

# -------------------------------
# 4️⃣ Chat logic with memory + logging
# -------------------------------
chat_memory = []  # stores past messages

LOG_FILE = "chat_log.csv"

def log_interaction(user_input, response, response_type):
    """Append chat logs to CSV for analysis"""
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), user_input, response, response_type])

def finance_chatbot(user_input):
    text = user_input.lower().strip()

    # 1. Domain check
    if not any(keyword in text for keyword in finance_keywords):
        response = "💡 I specialize in finance and investing topics. Please ask me about things like stocks, bonds, interest rates, or saving strategies."
        log_interaction(user_input, response, "domain_filter")
        return response

    # 2. Rule-based response
    for key, resp in predefined_responses.items():
        if key in text:
            chat_memory.append((user_input, resp))
            log_interaction(user_input, resp, "rule_based")
            return resp

    # 3. Model-based response with memory
    if chatbot:
        try:
            # Include last 2 turns of memory for context
            context = " ".join([f"User: {u} Bot: {b}" for u, b in chat_memory[-2:]])
            prompt = f"{context}\nUser: {user_input}\nBot:"

            model_output = chatbot(prompt, max_length=200, do_sample=True, temperature=0.3)[0]['generated_text']

            # Clean and validate model output
            if not re.search(r"(invest|stock|bond|money|interest|finance|fund|risk)", model_output.lower()):
                response = "⚠️ That seems slightly off-topic. Could you rephrase your question related to finance or investing?"
                log_interaction(user_input, response, "filtered_model")
                return response

            response = re.sub(r"[\n\r]+", " ", model_output).strip()
            chat_memory.append((user_input, response))
            log_interaction(user_input, response, "model")
            return response[:500]

        except Exception as e:
            response = f"⚠️ Model error: {str(e)}"
            log_interaction(user_input, response, "error")
            return response
    else:
        response = "🚫 The AI model isn't available right now. Please try again later."
        log_interaction(user_input, response, "no_model")
        return response

# -------------------------------
# 5️⃣ Gradio interface
# -------------------------------
title = "💬 Smart Finance Chatbot (Hybrid + Memory + Logging)"
description = """
Ask me anything about **investing, stocks, bonds, saving, or finance concepts.**  
I use both **rule-based logic** for quick answers and an **AI model** for flexible reasoning — with conversation memory and analytics.
"""

chatbot_ui = gr.ChatInterface(
    fn=finance_chatbot,
    title=title,
    description=description,
    theme="soft",
    examples=[
        "What is investing?",
        "Explain compound interest",
        "How do bonds work?",
        "What’s diversification in investing?",
        "Tell me about retirement savings plans"
    ],
)

if __name__ == "__main__":
    chatbot_ui.launch()
