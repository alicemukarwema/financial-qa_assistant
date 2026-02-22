"""
Financial Q&A Assistant - Gradio App for HuggingFace Spaces
Fine-tuned TinyLlama model for financial question answering

Author: mukarwema
Model: mukarwema/financial-qa-assistant
Dataset: gbharti/finance-alpaca
"""

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os


# MODEL LOADING


print("🚀 Loading Financial Q&A Assistant...")
print("=" * 80)

# Model configuration - Try fine-tuned first, fallback to base model
FINETUNED_MODEL = "mukarwema/financial-qa-assistant"
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🖥️  Device: {DEVICE}")

# Try to load fine-tuned model, fallback to base model if it fails
MODEL_NAME = None
tokenizer = None
model = None

# Attempt 1: Try fine-tuned model
try:
    print(f"📦 Attempting to load fine-tuned model: {FINETUNED_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    MODEL_NAME = FINETUNED_MODEL
    print(f"✅ Fine-tuned tokenizer loaded successfully!")
except Exception as e:
    print(f"⚠️  Could not load fine-tuned model: {str(e)[:100]}...")
    print(f"📦 Falling back to base model: {BASE_MODEL}")
    
    # Attempt 2: Load base model
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        MODEL_NAME = BASE_MODEL
        print(f"✅ Base model tokenizer loaded successfully!")
    except Exception as e2:
        print(f"❌ Error loading base model tokenizer: {e2}")
        raise

# Load model (matching the tokenizer we successfully loaded)
try:
    print(f"📥 Loading model weights for: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

print("=" * 80)
print(f"📊 Using model: {MODEL_NAME}")
print("=" * 80)
print(" Financial Q&A Assistant is ready!")
print("=" * 80)


# INFERENCE FUNCTION


def answer_financial_question(question):
    """
    Generate financial advice/answers using the fine-tuned model.
    
    Args:
        question (str): Financial question from user
        
    Returns:
        str: Generated answer
    """
    if not question.strip():
        return " Please enter a question."
    
    # Format the prompt
    question_text = f"### Question: {question}\n\n### Answer:"
    
    # Tokenize input
    inputs = tokenizer(
        question_text, 
        return_tensors="pt", 
        max_length=400, 
        truncation=True
    ).to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the answer part
    answer = response.split("### Answer:")[-1].strip()
    
    return answer


# GRADIO INTERFACE


# Example questions
EXAMPLES = [
    ["What is diversification in investing?"],
    ["How do interest rates affect bond prices?"],
    ["What is the difference between a stock and a bond?"],
    ["What factors should I consider before investing in stocks?"],
    ["How does compound interest work?"],
    ["What is a 401(k) retirement plan?"],
    ["What is an ETF and how does it work?"],
    ["How can I reduce my investment risk?"],
    ["What is the difference between a Roth IRA and a Traditional IRA?"],
    ["How do I start building an emergency fund?"]
]

# Create interface
demo = gr.Interface(
    fn=answer_financial_question,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Ask any financial question...",
        label=" Your Financial Question",
        info="Ask about investing, retirement, taxes, budgeting, and more!"
    ),
    outputs=gr.Textbox(
        label=" Answer",
        lines=10,
        show_copy_button=True
    ),
    title=" Financial Q&A Assistant",
    description=(
        " **AI-Powered Financial Advisor**\n\n"
        f"Using model: **{MODEL_NAME}**\n\n"
        "This assistant provides helpful, informative answers about finance, investing, banking, "
        "retirement planning, taxes, and more! Ask any financial question to get started.\n\n"
        " **Disclaimer**: This is an AI assistant for educational purposes only. "
        "Always consult with a qualified financial advisor for personalized advice."
    ),
    examples=EXAMPLES,
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
    ),
    cache_examples=False,
)


# LAUNCH


if __name__ == "__main__":
    print("\n Launching Gradio interface...")
    demo.launch(
        share=False,  # Set to True for temporary share link
        server_name="0.0.0.0",  # Important for HuggingFace Spaces
        server_port=7860,
    )
