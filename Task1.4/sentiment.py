# task4_sentiment.py

positive_words = ['love', 'happy', 'good', 'excellent', 'awesome']
negative_words = ['hate', 'sad', 'bad', 'horrible', 'terrible']
negations = ['not', "don't", "never", "no"]

def rule_based_sentiment(tokens):
    score = 0
    negation = False
    for word in tokens:
        if word in negations:
            negation = True
            continue
        if word in positive_words:
            score += -1 if negation else 1
        elif word in negative_words:
            score += 1 if negation else -1
        negation = False
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"

# Test example
if __name__ == "__main__":
    print("Sample Test:", rule_based_sentiment(['not', 'happy']))