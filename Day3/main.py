# main.py - Run all three models and display comparison

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import rule_based
import ml_model
import transformer

# ── Task 2: Single sentence demo ──────────────────────────────────────────────
TARGET = "I thought it would be bad, but it was amazing."

print("=" * 60)
print("TASK 2 — Pretrained Transformer Pipeline (BERT)")
print("=" * 60)
result = transformer.analyze(TARGET)
print(f'Input : "{TARGET}"')
print(f"Label : {result['label']}")
print(f"Confidence: {result['confidence']:.2f}")

# ── Task 3: Comparison table ───────────────────────────────────────────────────
TEST_SENTENCES = [
    "I thought it would be bad, but it was amazing.",
    "This is the worst experience I have ever had.",
    "The product is okay, nothing special.",
    "Absolutely fantastic! I love every bit of it.",
    "Terrible quality, complete waste of money.",
]

print("\n" + "=" * 60)
print("TASK 3 — Comparison Table: Rule-Based vs ML vs Transformer")
print("=" * 60)

# Header
col_w = 18
h_sent = 46
print(f"{'Sentence':<{h_sent}} {'Rule-Based':<{col_w}} {'ML Model':<{col_w}} {'Transformer':<{col_w}}")
print("-" * (h_sent + col_w * 3))

for sentence in TEST_SENTENCES:
    rb  = rule_based.analyze(sentence)
    ml  = ml_model.analyze(sentence)
    tr  = transformer.analyze(sentence)

    rb_str  = f"{rb['label']} ({rb['confidence']:.2f})"
    ml_str  = f"{ml['label']} ({ml['confidence']:.2f})"
    tr_str  = f"{tr['label']} ({tr['confidence']:.2f})"

    # Wrap long sentences
    display = sentence if len(sentence) <= h_sent else sentence[:h_sent-3] + "..."
    print(f"{display:<{h_sent}} {rb_str:<{col_w}} {ml_str:<{col_w}} {tr_str:<{col_w}}")

print("=" * 60)
print("Done.")