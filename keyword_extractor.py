# keyword_extractor.py
import pke

def extract_keywords(text):
    print("Keywords")
    extractor = pke.unsupervised.MultipartiteRank()
    extractor.load_document(input=text, language='en')
    extractor.candidate_selection()
    extractor.candidate_weighting()
    keyphrases = extractor.get_n_best(n=15)
    return [keyphrase[0] for keyphrase in keyphrases]
