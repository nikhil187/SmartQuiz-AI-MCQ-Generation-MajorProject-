import sys
sys.path.append(r"C:\Users\laksh\Downloads\bert-extractive-summarizer-master")

from summarizer import Summarizer

def summarize_text(text, model_name="bert-base-uncased", max_chunk_sentences=30):
    model = Summarizer(model=model_name)
    sentences = text.split('.')  # Split text into sentences
    chunks = []
    chunk = []
    summary_chunks = []

    # Divide text into chunks of max_chunk_sentences sentences
    for sentence in sentences:
        chunk.append(sentence.strip())
        if len(chunk) >= max_chunk_sentences:
            chunks.append('. '.join(chunk))
            chunk = []

    if chunk:
        chunks.append('. '.join(chunk))

    # Summarize each chunk and combine summaries
    for chunk in chunks:
        summary = model(chunk)
        summary_chunks.append(summary)

    combined_summary = ' '.join(summary_chunks)
    return combined_summary
