import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import os
import random
import time

# Load environment variables
load_dotenv()

print("Configuring the Gemini API...")
try:
    # Configure the Gemini API
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file.")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    print("Gemini API configured successfully.")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    exit()


# Load the generated title pairs
try:
    df = pd.read_csv('data/idea-title_pairs.csv')
    print(f"Loaded {len(df)} total title pairs.")
except FileNotFoundError:
    print("Error: data/idea-title_pairs.csv not found. Please run 'scripts/01_generate_titles.py' first.")
    exit()

# --- Simplified Logic: Process only the first 30 pairs ---
NUM_PAIRS_TO_PROCESS = 30
if len(df) < NUM_PAIRS_TO_PROCESS:
    NUM_PAIRS_TO_PROCESS = len(df) # Adjust if the file has fewer than 30 pairs

df_to_process = df.head(NUM_PAIRS_TO_PROCESS).copy()
print(f"Processing the first {len(df_to_process)} pairs to create a sample preference file.")


def get_gemini_preference(idea, title_a, title_b):
    """
    Uses Gemini to determine the more engaging YouTube title.
    Returns 1 if title_b is preferred, 0 if title_a is preferred.
    """
    prompt = f"""You are an expert in creating viral YouTube titles.
Given a video idea and two potential titles, which one is more engaging and likely to get more clicks?

**Video Idea**: {idea}
**Title A**: {title_a}
**Title B**: {title_b}

Your response must be a single character: 'A' or 'B'.
"""
    try:
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().upper()
        if 'B' in cleaned_response:
            return 1
        return 0
    except Exception as e:
        print(f"  - An error occurred: {e}. Defaulting to random choice.")
        return random.choice([0, 1])

preferences = []
total_pairs = len(df_to_process)
print("Generating preferences using Gemini...")

for index, row in df_to_process.iterrows():
    print(f"  - Processing pair {len(preferences) + 1}/{total_pairs}...")
    preference = get_gemini_preference(row['idea'], row['title_a'], row['title_b'])
    preferences.append(preference)
    # Sleep for over 2 seconds to be well below any free-tier rate limit.
    time.sleep(2.1)

# Add the new column to the dataframe
df_to_process['title_b_preferred'] = preferences

# Save the processed pairs to the preferences file
df_to_process.to_csv('data/idea-title_pairs-preferences.csv', index=False)

print(f"\nSuccessfully generated {len(df_to_process)} preferences and saved them!")
print(df_to_process.head())