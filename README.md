# Selfcheckgpt-scoring
Code
!pip install selfcheckgpt pandas tqdm matplotlib seaborn scikit-learn reportlab

from google.colab import files
uploaded = files.upload()

# Example: if your file is wikibio_gpt3_dataset.csv
import pandas as pd

# Replace with your file name from upload
INPUT_PATH = "wikibio_gpt3_dataset (1).csv"
# Added on_bad_lines='skip' to try and handle parsing errors
df = pd.read_csv(INPUT_PATH, encoding="utf-8", low_memory=False, on_bad_lines='skip')
print("✅ File loaded successfully!")
print("Rows:", len(df))
print("Columns:", list(df.columns))
display(df.head(3))

import pandas as pd
df = pd.read_csv("selfcheck_output.csv", low_memory=False)
print("Columns:", df.columns.tolist())
print("Status counts:")
print(df["status"].value_counts(dropna=False))
print("Non-null avg_score:", df["avg_score"].notna().sum())
print(df[["gpt3_text", "wiki_bio_text", "gpt3_text_samples"]].head(3))


# ✅ Full working SelfCheckGPT scoring pipeline (with safe parsing)

!pip install selfcheckgpt tqdm pandas transformers --quiet

import pandas as pd
import ast
import json
from tqdm import tqdm
from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore

# ========== STEP 1: Load your CSV ==========
INPUT_PATH = "selfcheck_output.csv"   # change if your file name differs
df = pd.read_csv(INPUT_PATH)
print("✅ File loaded:", INPUT_PATH)
print("Columns:", df.columns.tolist())
print("Rows:", len(df))

# ========== STEP 2: Safe parser for messy list strings ==========
def safe_parse_list(s):
    """Safely parse JSON-like or Python-list-like text into a Python list."""
    if pd.isna(s):
        return []
    s = str(s).replace("\n", " ").strip()
    try:
        return json.loads(s)
    except:
        try:
            return ast.literal_eval(s)
        except:
            return []

# ========== STEP 3: Clean and inspect a test sample ==========
test_row = df.iloc[0]
gpt3_sentences = safe_parse_list(test_row["gpt3_sentences"])
gpt3_samples = safe_parse_list(test_row["gpt3_text_samples"])

print("\n🧩 Sample Parsed:")
print("gpt3_sentences:", gpt3_sentences[:2])
print("gpt3_samples:", gpt3_samples[:2])

# ========== STEP 4: Initialize SelfCheck-BERTScore ==========
selfcheck_scorer = SelfCheckBERTScore()
print("\n✅ SelfCheck-BERTScore initialized")

# ========== STEP 5: Define scoring function ==========
def calculate_scores(row):
    try:
        gpt3_sentences = safe_parse_list(row["gpt3_sentences"])
        gpt3_samples = safe_parse_list(row["gpt3_text_samples"])

        if not gpt3_sentences or not gpt3_samples:
            return None, [], "skipped: empty sentences or samples"

        # Call the model correctly (without invalid arguments)
        scores = selfcheck_scorer.predict(
            sentences=gpt3_sentences,
            sampled_passages=gpt3_samples
        )

        # Handle different return types
        if isinstance(scores, dict):
            avg_score = scores.get("manager_score") or scores.get("score") or None
            sent_scores = scores.get("sentence_scores") or []
        elif isinstance(scores, (list, float)):
            avg_score = sum(scores) / len(scores) if isinstance(scores, list) else scores
            sent_scores = scores
        else:
            avg_score = None
            sent_scores = []

        return avg_score, sent_scores, "success"

    except Exception as e:
        return None, [], f"error: {e}"

# ========== STEP 6: Run scoring on all rows ==========
tqdm.pandas(desc="Scoring rows (safe)")
df[["avg_score", "sent_scores", "status"]] = df.progress_apply(calculate_scores, axis=1, result_type='expand')

print("\n✅ Scoring complete!")
print("Status counts:")
print(df["status"].value_counts(dropna=False))
print("\nNon-null avg_score count:", df["avg_score"].notna().sum())

# ========== STEP 7: Save output ==========
df.to_csv("selfcheck_scored_output.csv", index=False)
print("\n💾 Saved scored file: selfcheck_scored_output.csv")

# ========== STEP 8: Inspect results ==========
print("\n📉 Top 5 lowest-scoring examples:")
print(df.sort_values("avg_score").head(5)[["gpt3_text", "avg_score", "status"]].to_string(index=False))
