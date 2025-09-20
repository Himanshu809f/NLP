import spacy

nlp = spacy.load("en_core_web_sm")

text="Apple's runner was running quickly words, but it is the best runner."
doc = nlp(text)
for token in doc:
    print(token.text,"->",token.lemma_,"stop?",token.is_stop)