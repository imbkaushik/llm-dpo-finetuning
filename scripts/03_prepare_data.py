import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict

print("Loading preference data...")
try:
    df = pd.read_csv('data/idea-title_pairs-preferences.csv')
    print(f"Loaded {len(df)} preferences.")
except FileNotFoundError:
    print("Error: 'data/idea-title_pairs-preferences.csv' not found.")
    print("Please run 'scripts/02_generate_preferences_gemini.py' first.")
    exit()

# --- Create Prompt ---
template = lambda idea: f"""Given the YouTube video idea write an engaging title.

**Video Idea**: {idea}

**Additional Guidance**:
- Title should be between 30 and 75 characters long
- Only return the title idea, nothing else!"""

def idea_to_prompt(idea):
    return [{"role": "user", "content": template(idea.lower())}]

df['prompt'] = df['idea'].apply(idea_to_prompt)
print("Formatted prompts created.")

# --- Create Chosen and Rejected Responses ---
def title_to_completion(title):
    return [{"role": "assistant", "content": title}]

# Create 'chosen' and 'rejected' columns based on the preference
df['chosen'] = np.where(
    df['title_b_preferred'] == 1, 
    df['title_b'].apply(title_to_completion), 
    df['title_a'].apply(title_to_completion)
)
df['rejected'] = np.where(
    df['title_b_preferred'] == 1, 
    df['title_a'].apply(title_to_completion), 
    df['title_b'].apply(title_to_completion)
)
print("Created 'chosen' and 'rejected' columns.")

# --- Create Train-Validation Split and Push to Hub ---
# Shuffle dataframe
df_shuffled = df[['prompt', 'chosen', 'rejected']].sample(frac=1, random_state=42).reset_index(drop=True)

# 90-10 split
train_size = int(0.9 * len(df_shuffled))

# Slice accordingly
df_train = df_shuffled.iloc[:train_size]
df_valid = df_shuffled.iloc[train_size:]

print(f"Data split into {len(df_train)} training samples and {len(df_valid)} validation samples.")

# Convert pandas DataFrames to Hugging Face Datasets
train_ds = Dataset.from_pandas(df_train)
valid_ds = Dataset.from_pandas(df_valid)

# Combine into a DatasetDict
dataset_dict = DatasetDict({
    'train': train_ds,
    'valid': valid_ds,
})

print("Dataset created. Pushing to Hugging Face Hub...")
try:
    hf_username = "imbkaushik" 
    dataset_dict.push_to_hub(f"{hf_username}/youtube-titles-dpo")
    print(f"Dataset successfully pushed to Hugging Face Hub at '{hf_username}/youtube-titles-dpo'.")
except Exception as e:
    print(f"Error pushing to Hub: {e}")
    print("Please ensure you are logged in via 'huggingface-cli login'.")