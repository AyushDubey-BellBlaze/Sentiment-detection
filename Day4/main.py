# main.py — Day 4: Run all 3 tasks + Interactive Emotion Detector
import subprocess, sys

print("\n" + "🔵 " * 20)
print("DAY 4 — NLP with Transformers")
print("🔵 " * 20 + "\n")

for task, file in [
    ("Task 1: Load IMDB Dataset",      "task1_load_dataset.py"),
    ("Task 2: Fine-Tune BERT",         "task2_finetune.py"),
    ("Task 3: Multi-Emotion Detection","task3_emotion.py"),
]:
    print(f"\n{'='*55}")
    print(f"  {task}")
    print(f"{'='*55}")
    subprocess.run([sys.executable, file])

# ── Interactive Mode ─────────────────────────────────────────
print("\n" + "="*55)
print("  🎭 Interactive Emotion Detector")
print("="*55)
print("  Would you like to test your own sentences?")

choice = input("\n  Enter 'yes' to start interactive mode (or 'no' to exit): ").strip().lower()

if choice in ("yes", "y"):
    subprocess.run([sys.executable, "interactive_emotion.py"])
else:
    print("\n👋 All tasks complete! .")