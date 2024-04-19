# distractor_generator.py
from sense2vec import Sense2Vec
import string
from collections import OrderedDict
import google.generativeai as genai

def edits(word):
    "All edits that are one edit away from `word`."
    letters = 'abcdefghijklmnopqrstuvwxyz ' + string.punctuation
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def generate_distractors_with_gemini(word, context, question):
    # Configure Gemini API key

    GOOGLE_API_KEY = "AIzaSyB_5-xr_i0PIgg7cKFi5okUgixkuB-LQIY"
    genai.configure(api_key=GOOGLE_API_KEY)

    # Construct a prompt using the provided context
    prompt = f"Generate plausible distractors related to the context and answer, for the answer:'{word}' in the following context: {context} where the question is:{question} to create MCQS.Where give in this format only:A) B) C) D) E) F)"

    model = genai.GenerativeModel('gemini-pro')  # Or your chosen Gemini model
    chat = model.start_chat(history=[])

    response = chat.send_message(prompt)
    string = response.text
    print(string)
    # Extract distractors (modify if Gemini's response format is different)
    distractors = [item.strip() for item in response.text.split(',')]
    string = distractors[0]

    # Convert string to list
    distractors_list = [item.strip().lstrip('- ') for item in string.split('\n') if item.strip()]

    # Truncate the first 4 characters from each item in the list
    distractors_list_truncated = [item[3:] for item in distractors_list]
    return distractors_list_truncated


def sense2vec_get_words(word, s2v):
    output = []

    word_preprocessed = word.translate(str.maketrans("", "", string.punctuation))
    word_preprocessed = word_preprocessed.lower()

    word_edits = edits(word_preprocessed)

    word = word.replace(" ", "_")

    try:
        sense = s2v.get_best_sense(word)
        most_similar = s2v.most_similar(sense, n=15)

        compare_list = [word_preprocessed]
        for each_word in most_similar:
            append_word = each_word[0].split("|")[0].replace("_", " ")
            append_word = append_word.strip()
            append_word_processed = append_word.lower()
            append_word_processed = append_word_processed.translate(
                str.maketrans("", "", string.punctuation))
            if append_word_processed not in compare_list and word_preprocessed not in append_word_processed and append_word_processed not in word_edits:
                output.append(append_word.title())
                compare_list.append(append_word_processed)
        out = list(OrderedDict.fromkeys(output))
        return out
    except Exception as e:
        print("Sense2vec_distractors failed for word:", word)
        print(e)
        return []

def get_options(answer,sentence,question,s2v):
    print("Distractor generation")
    distractors = []

    # Attempt to generate distractors using Gemini API
    try:
        distractors = generate_distractors_with_gemini(answer,sentence,question)
        if distractors:
            return distractors, "Gemini"

    except Exception as e:  # Error using Gemini
        print(f"Error using Gemini API: {e}")

    # Fallback to sense2vec
    try:
        distractors = get_sense2vec_distractors(answer)
        if distractors:
            return distractors, "sense2vec"
    except Exception as e:
        print(f"Error using sense2vec: {e}")

    return distractors, "None"  # No distractors found
