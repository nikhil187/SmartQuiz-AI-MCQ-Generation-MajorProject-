# mcq_generator.py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def generate_question_and_answer(context, tokenizer, model):
    print("Q and A generation")
    inputs = tokenizer(context, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    question_answer = tokenizer.decode(outputs[0], skip_special_tokens=False)
    question_answer = question_answer.replace(tokenizer.pad_token, "").replace(tokenizer.eos_token, "")
    question, answer = question_answer.split(tokenizer.sep_token)
    return question, answer
