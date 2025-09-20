import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Sample text
text = "Apple's runner was running quickly words, but it is the best runner."

# Tokenization
tokens = word_tokenize(text)
print("Tokens:", tokens)


sw=set(stopwords.words('english'))
tokens_nsw=[t for t in tokens if t.lower() not in sw]   
print("Tokens without Stopwords:", tokens_nsw  )

# stemming
stemmer=PorterStemmer()
print("Stemmed Tokens:", [stemmer.stem(t) for t in tokens_nsw])

# Lemmatization
lemmatizer=WordNetLemmatizer()
print("Lemmatized Tokens:", [lemmatizer.lemmatize(t) for t in tokens_nsw])

