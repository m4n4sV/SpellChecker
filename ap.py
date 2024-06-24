import streamlit as st
import torch
from transformers import BertTokenizer, BertForMaskedLM
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import words
from spellchecker import SpellChecker

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('words')

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

# Set of English words for basic spell checking
english_vocab = set(words.words())

# Initialize SpellChecker
spell = SpellChecker()

def tokenize_sentence(sentence):
    tokens = word_tokenize(sentence)
    return tokens

def correct_spelling_with_bert(tokens):
    corrected_tokens = tokens.copy()

    for i, token in enumerate(tokens):
        if token.lower() not in english_vocab:
            # Mask the token
            tokens_copy = tokens.copy()
            tokens_copy[i] = '[MASK]'

            # Convert tokens to IDs
            token_ids = tokenizer.convert_tokens_to_ids(tokens_copy)

            # Skip if token_ids contains None
            if None in token_ids:
                continue

            token_tensor = torch.tensor([token_ids])

            # Predict all tokens
            with torch.no_grad():
                outputs = model(token_tensor)
                predictions = outputs[0]

            predicted_index = torch.argmax(predictions[0, i]).item()
            predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
            corrected_tokens[i] = predicted_token

    return corrected_tokens

def spell_check_and_correct(sentence):
    tokens = tokenize_sentence(sentence)
    corrected_tokens = []
    misspelled_words = []

    for token in tokens:
        corrected_word = spell.correction(token)
        if corrected_word and corrected_word != token:
            misspelled_words.append(token)
        corrected_tokens.append(corrected_word if corrected_word else token)

    # Further correct tokens using BERT for context
    corrected_tokens = correct_spelling_with_bert(corrected_tokens)

    # Construct the corrected sentence
    corrected_sentence = ' '.join(corrected_tokens)

    return misspelled_words, corrected_sentence

# Streamlit UI
st.title("📝 Spelling Checker")
st.markdown("# Welcome to the Spelling Checker! ✨📚")

user_input = st.text_area("Enter a sentence:", height=200)
if st.button("Check Spelling"):
    misspelled_words, corrected_sentence = spell_check_and_correct(user_input)

    if misspelled_words:
        st.write(f"*Original:* {user_input}")
        st.write(f"*Misspelled words:* {misspelled_words}")
        st.write(f"*Corrected:* {corrected_sentence}")
    else:
        st.write(f"*Original:* {user_input}")
        st.write(f"No misspelled words found. *Corrected:* {corrected_sentence}")
