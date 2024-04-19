import streamlit as st
from summarizing import summarize_text
from keyword_extractor import extract_keywords
from sentence_processor import extract_sentences_with_keywords,filter_sentences
from distractor_generator import generate_distractors_with_gemini, get_options
from mcq_generator import generate_question_and_answer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sense2vec import Sense2Vec
import random

st.title('Generate MCQs from Text')

original_text = st.text_area('Enter your text here:', """The sun is the largest star in our solar system. It is present in the centre of the earth and the planets orbit around the sun. The sun is spherical in shape and scientists state that it contains a mass of hot plasma. It is essential for our planet earth as it gives us the energy which we require for the existence of life. Through the essay on sun, we will go through the details and their importance.""")

if st.button('Generate MCQs'):
    
    summary = summarize_text(original_text)
    keywords_original = extract_keywords(original_text)
    keywords_summarized = extract_keywords(summary)
    common_keywords = set(keywords_original).intersection(set(keywords_summarized))
    print(len(keywords_original))
    print(len(keywords_summarized))
    print(len(common_keywords))
    extracted_sentences = extract_sentences_with_keywords(original_text, common_keywords)
    print(len(extracted_sentences))
    fs = []
    batch_size = 10
    extracted_sentence_count = len(extracted_sentences)

    for i in range(0, extracted_sentence_count, batch_size):
        batch_sentences = extracted_sentences[i:i+batch_size]
        filtered_sentences = filter_sentences(batch_sentences)
        fs.extend(filtered_sentences)

    print(len(fs)) 
    model_path = 'potsawee/t5-large-generation-squad-QuestionAnswer'
    tokenizer_path = 'potsawee/t5-large-generation-squad-QuestionAnswer'

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    s2v_model_path = r'C:\Users\laksh\Downloads\s2v_old'
    s2v = Sense2Vec().from_disk(s2v_model_path)

    quiz_questions = []
    for sentence in fs:
        context = sentence  # Use the sentence as the context
        # Generate question and answer
        question, answer = generate_question_and_answer(context, tokenizer, model)
        question = f"MCQ: {question}"  # Format the question as MCQ
        correct_option = answer[0].upper() + answer[1:].strip()
        distractors = get_options(answer, sentence, question, s2v)
        dis = []
        if len(distractors[0]) > 3:
            x = 3
        elif len(distractors[0]) == 0:
            x = 0
        else:
            x = len(distractors[0])
        for i in range(0, x):
            dis.append(distractors[0][i])
        cleaned_options = [correct_option] + dis
        random.shuffle(cleaned_options)
        print("context:" + context)
        print("question:" + question)
        for i, option in enumerate(cleaned_options):
            print(f"{chr(65 + i)}. {option}")
        print()
        quiz_questions.append({
            "context:": context,
            "question": question,
            "options": cleaned_options,
            "correct_answer": correct_option
    })

    for i, question in enumerate(quiz_questions, start=1):
        st.write(f"**Question {i}:** {question['question']}")
        for j, option in enumerate(question['options']):
            st.write(f"   {chr(65 + j)}. {option}")
        st.write(f"   Correct Answer: {chr(65 + question['options'].index(question['correct_answer']))}")
        st.write("")  # Add an empty line between questions for better readability
