import csv
import re
from itertools import combinations
from together import Together
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

print("Setting up Together AI client...")
# Set Together API key
try:
    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    print("Client created successfully.")
except Exception as e:
    print(f"Error creating Together client: {e}")
    exit()


# --- Load Ideas from CSV ---
idea_list = []
try:
    with open('data/ideas.csv', mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            idea_list.append(row[0])
    print(f"Loaded {len(idea_list)} ideas from data/ideas.csv")
except FileNotFoundError:
    print("Error: data/ideas.csv not found. Please create it and add your video ideas.")
    exit()


# --- Prompt Template ---
# This template provides examples to the LLM for few-shot prompting.
template = lambda idea: f"""**YouTube Titles**:
- The 8 AI Skills That Will Separate Winners From Losers in 2025
- World's Lightest Solid!
- Why Are 96,000,000 Black Balls on This Reservoir?
- I Make $15K/Month With 2 AI Apps
- How I Would Become a Data Analyst if I had to Start Over
- 6 Years of Studying Machine Learning in 26 Minutes
- The Complete Machine Learning Roadmap
- AI Explained at 5 Levels of Complexity
--
Given the YouTube video idea, write 5 engaging title ideas.

**Video Idea**: {idea}

**Additional Guidance**:
- Titles should be between 30 and 75 characters long.
- Only return the title ideas, nothing else!
- Title ideas should be written as an ordered markdown list.
"""

# --- Generate Titles and Create Pairs ---
triplet_list = []
print("Generating titles for each idea...")
for idea in idea_list:
    print(f"  - Processing idea: '{idea}'")
    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct-Turbo",
            messages=[{"role": "user", "content": template(idea)}],
            temperature=0.7,
            top_p=0.7,
        )
        response_raw = response.choices[0].message.content
        
        # Parse the completion to extract the 5 titles
        pattern = r"^\s*(?:[-*]|\d+\.)\s+(.+)$"
        title_list = re.findall(pattern, response_raw, re.MULTILINE)

        if len(title_list) >= 2:
            # Generate all possible unique pairs of titles
            title_pair_list = list(combinations(title_list, 2))
            
            # Store all unique idea-title pairs
            for a, b in title_pair_list:
                triplet_list.append({"idea": idea, "title_a": a, "title_b": b})
        else:
            print(f"    - Warning: Could not parse enough titles for '{idea}'. Received: {response_raw}")

    except Exception as e:
        print(f"    - Error generating titles for '{idea}': {e}")

# --- Write Pairs to CSV ---
if triplet_list:
    print(f"Generated {len(triplet_list)} title pairs.")
    try:
        with open("data/idea-title_pairs.csv", mode="w", newline="", encoding="utf-8") as file:
            fieldnames = ["idea", "title_a", "title_b"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(triplet_list)
        print("Successfully saved title pairs to data/idea-title_pairs.csv")
    except Exception as e:
        print(f"Error writing to CSV: {e}")
else:
    print("No title pairs were generated. Exiting.")