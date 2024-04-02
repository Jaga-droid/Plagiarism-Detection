import nltk
from nltk.tokenize import SyllableTokenizer
nltk.download('punkt')
def Reading_Grade_Level(text):

    # Tokenizing text into words, sentences and syllables
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
    SSP = SyllableTokenizer()
    syllables = [len(SSP.tokenize(word)) for word in words]
    
    # Calculating average number of words for each sentence
    words_per_sentence = len(words)/len(sentences)

    # Calculating average syllables for each word
    syllables_per_word = sum(syllables)/len(words)

    # Using the Flesh-Kincaid formula for readability of text
    readability_score = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
    
    return readability_score
