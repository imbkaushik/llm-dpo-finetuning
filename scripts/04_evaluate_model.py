import pandas as pd
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

print("--- Model Evaluation Script ---")

# --- Configuration ---
HF_USERNAME = "imbkaushik"
FT_MODEL_NAME = "Qwen2.5-0.5B-DPO-YouTube-Titles"
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
YOUR_FT_MODEL_ID = f"{HF_USERNAME}/{FT_MODEL_NAME}"
NUM_IDEAS_TO_TEST = 10
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# --- Prompt Template ---
def format_prompt(idea):
    return f"""Given the YouTube video idea, write an engaging title.

**Video Idea**: {idea}

**Additional Guidance**:
- Title should be between 30 and 75 characters long
- Only return the title idea, nothing else!
"""

# --- Title Generation Function ---
def generate_response(prompt, model, tokenizer, max_new_tokens=60, temperature=0.7, device=DEVICE):
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device in ['cuda', 'mps'] else -1)
    outputs = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        num_return_sequences=1,
        truncation=True
    )
    # Return the part after the instruction to extract only the title
    return outputs[0]['generated_text'].split('nothing else!')[-1].strip()

# --- Load Test Ideas ---
try:
    df_ideas = pd.read_csv('data/ideas.csv', header=None)
    idea_list = df_ideas[0].tolist()

    if idea_list[0].strip().lower() == 'idea':
        idea_list.pop(0)

    test_ideas = random.sample(idea_list, min(NUM_IDEAS_TO_TEST, len(idea_list)))
    print(f"Loaded {len(test_ideas)} ideas for evaluation.")
except FileNotFoundError:
    print("Error: 'data/ideas.csv' not found. Please ensure the file exists.")
    exit()

# --- Load Models ---
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

print("Loading fine-tuned model...")
try:
    ft_model = AutoModelForCausalLM.from_pretrained(YOUR_FT_MODEL_ID)
    ft_tokenizer = AutoTokenizer.from_pretrained(YOUR_FT_MODEL_ID)
except Exception as e:
    print(f"Error loading fine-tuned model '{YOUR_FT_MODEL_ID}': {e}")
    exit()

# --- Run Evaluation ---
results = []
print("\nGenerating titles for comparison...")
for i, idea in enumerate(test_ideas):
    print(f"  - Processing idea {i+1}/{len(test_ideas)}: '{idea}'")

    prompt = format_prompt(idea)
    base_title = generate_response(prompt, base_model, base_tokenizer)
    ft_title = generate_response(prompt, ft_model, ft_tokenizer)

    results.append({
        "idea": idea,
        "base_model_title": base_title,
        "fine_tuned_title": ft_title
    })

# --- Display and Save Results ---
df_results = pd.DataFrame(results)
print("\n--- Evaluation Results ---")
print(df_results)

output_path = "data/evaluation_results.csv"
df_results.to_csv(output_path, index=False)
print(f"\nResults saved to '{output_path}'.")
