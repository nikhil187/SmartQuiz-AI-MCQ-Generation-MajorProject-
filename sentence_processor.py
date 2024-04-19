# sentence_processor.py
from flashtext import KeywordProcessor
import nltk
from nltk.tokenize import sent_tokenize
import google.generativeai as genai
def filter_sentences(sen):
    """
    Generates plausible distractors for a given word using Gemini.

    Args:
        word (str): The target word.
        context (str): The context in which the target word is used.

    Returns:
        list: A list of distractors.
    """
    GOOGLE_API_KEY = "AIzaSyB_5-xr_i0PIgg7cKFi5okUgixkuB-LQIY"
    genai.configure(api_key=GOOGLE_API_KEY)
    # Construct a prompt using the provided context
    prompt = f"Filter the following sentences to ensure to create best questions from the sentences just give me the reply which are the best sentences(condition:*you dont frame any question just give me the sentences from which i gave):{sen}"

    model = genai.GenerativeModel('gemini-pro')  # Or your chosen Gemini model
    chat = model.start_chat(history=[])

    response = chat.send_message(prompt)
    string=response.text
    filter_list = [item.strip().lstrip('- ') for item in string.split('\n') if item.strip()]
    return filter_list

def extract_sentences_with_keywords(text, keywords):
    print("Extracting Sentences")
    keyword_processor = KeywordProcessor()
    for keyword in keywords:
        keyword_processor.add_keyword(keyword)

    sentences = sent_tokenize(text)
    keyword_sentences = set()
    for sentence in sentences:
        if keyword_processor.extract_keywords(sentence):
            keyword_sentences.add(sentence.strip())

    return list(keyword_sentences)
