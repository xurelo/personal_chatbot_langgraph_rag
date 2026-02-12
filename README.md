Personal chat bot assistant for CV using RAG and Langgraph/Langchain

This is a personal project of Generative AI to answer to questions on a CV (or other documents)

# Features
- Chatbot to ask questions
- Audio recording to transcribe any recorded questions using the microphone

# Frameworks and software used
- Python Language
- Langchain/Langgraph
- Chroma
- Embedding through HuggingFace model sentence-transformers/all-MiniLM-L6-v2  (runs locally)
- Audio transcription using transformers pipeline with model openai/whisper-base. The transcription of the audio runs locally so it might be slow.

# Requirements
- Requires installing ffmpeg library for transcribing audio