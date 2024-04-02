import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
def detect_hyperbole(text):
    number_of_sentences=len(text.split('.'))
    tokens = word_tokenize(text)
    parts_of_speech = pos_tag(tokens)
    adjectives = [word for word,pos in parts_of_speech if pos in ['JJ','JJR','JJS']]
    return len(adjectives)/number_of_sentences