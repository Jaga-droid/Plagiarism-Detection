import spacy
from collections import Counter

nlp = spacy.load('en_core_web_lg')

def style_similarity(text1, text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)

    # POS tag proportions
    pos_counts1 = doc1.count_by(spacy.attrs.POS)
    pos_counts2 = doc2.count_by(spacy.attrs.POS)

    pos_proportions1 = {doc1.vocab[pos_id].text: count/max(len(doc1),1) for pos_id, count in pos_counts1.items()}
    pos_proportions2 = {doc2.vocab[pos_id].text: count/max(len(doc2),1) for pos_id, count in pos_counts2.items()}

    intersecting_pos = set(pos_proportions1.keys()) & set(pos_proportions2.keys())
    
    pos_similarity_score = sum(min(pos_proportions1[pos], pos_proportions2[pos]) for pos in intersecting_pos) / max(sum(pos_proportions1.values()), sum(pos_proportions2.values()),1)

    # Punctuation usage
    punct_counts1 = Counter(token.text for token in doc1 if token.is_punct)
    punct_counts2 = Counter(token.text for token in doc2 if token.is_punct)

    intersecting_punct = set(punct_counts1.keys()) & set(punct_counts2.keys())
    
    punct_similarity_score = sum(min(punct_counts1[punct], punct_counts2[punct]) for punct in intersecting_punct) / max(sum(punct_counts1.values()), sum(punct_counts2.values()),1)

    # Average sentence length
    avg_sent_len1 = sum(len(sent) for sent in doc1.sents) / max(len(list(doc1.sents)),1)
    avg_sent_len2 = sum(len(sent) for sent in doc2.sents) / max(len(list(doc2.sents)),1)

    sent_len_similarity_score = min(avg_sent_len1, avg_sent_len2) / max(avg_sent_len1, avg_sent_len2,1)

    # Use of stop words
    stop_counts1 = Counter(token.text for token in doc1 if token.is_stop)
    stop_counts2 = Counter(token.text for token in doc2 if token.is_stop)

    intersecting_stop = set(stop_counts1.keys()) & set(stop_counts2.keys())
    
    stop_similarity_score = sum(min(stop_counts1[stop], stop_counts2[stop]) for stop in intersecting_stop) / max(sum(stop_counts1.values()), sum(stop_counts2.values()),1)

    # Combine all scores
    total_score = (pos_similarity_score + punct_similarity_score + sent_len_similarity_score + stop_similarity_score) / 4

    return total_score * 100  # Return as percentage
