import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download once (comment after first run)
nltk.download('punkt')
nltk.download('stopwords')

# Input text
text = "I love this product!!!"

# 1. Lowercase
text = text.lower()

# 2. Remove punctuation
text = text.translate(str.maketrans('', '', string.punctuation))

# 3. Tokenization
tokens = word_tokenize(text)

# 4. Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in tokens if word not in stop_words]

# Output
print(filtered_words)
print("\nPreprocessing complete ✅")