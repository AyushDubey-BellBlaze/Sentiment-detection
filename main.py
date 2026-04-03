import nltk
nltk.download('punkt_tab')   # download required resource
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
import string

text = "I love this product!!!"
text = text.lower()
text = text.translate(str.maketrans('', '', string.punctuation))
tokens = text.split()
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in tokens if word not in stop_words]
print(filtered_words)