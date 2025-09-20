import ssl
import certifi
import nltk

# ✅ SSL verify error bypass
ssl._create_default_https_context = ssl._create_unverified_context

# ✅ certifi के certificates का path confirm (optional)
print("Using certificates from:", certifi.where())

# ✅ ज़रूरी NLTK resources download करें
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

print("\n✅ All NLTK data downloaded successfully!")
