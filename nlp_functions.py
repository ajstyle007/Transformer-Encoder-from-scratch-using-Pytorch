from transformers import DistilBertTokenizerFast, DistilBertModel
import torch
import torch.nn as nn
from encoder_layer import Encoder_block
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import nltk
from nltk.corpus import stopwords
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
bert_emb = DistilBertModel.from_pretrained("distilbert-base-uncased").get_input_embeddings().to(device)

bert_emb = bert_emb.to(device) 

d_model = 512
num_heads = 8
d_ff = 2048

projection = nn.Linear(768, 512).to(device)

encoder = Encoder_block(d_model=d_model, num_heads=num_heads, d_ff=d_ff).to(device)
classifier = nn.Linear(512, 3).to(device)

ckpt = torch.load("restored_model.pt", map_location=device)

projection.load_state_dict(ckpt["projection_state_dict"])
encoder.load_state_dict(ckpt["encoder_state_dict"])
classifier.load_state_dict(ckpt["classifier_state_dict"])

# ---------------------
# Keyword Extraction
# ---------------------

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
ADDITIONAL_STOP = {"(", ")", "-", ".", ",", "this", "with"}  # avoid duplicates like "to", "is" already in nltk
STOPWORDS = stop_words.union(ADDITIONAL_STOP)

def extract_keyphrases(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        x = bert_emb(input_ids)
        x = projection(x)
        enc_out, attn = encoder(x, mask=attention_mask)
        logits = classifier(enc_out)
        preds = torch.argmax(logits, dim=-1)[0].cpu().numpy()

    # Get tokens and subword information
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    phrases = []
    current_phrase_tokens = []   # will store clean full words (not subwords)

    for token, label in zip(tokens, preds):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        # Reconstruct real word from subword
        if token.startswith("##"):
            token = token[2:]  # remove ##
            if current_phrase_tokens:
                # append to last word being built
                current_phrase_tokens[-1] += token
            continue  # subword, don't decide B/I/O yet
        else:
            cleaned_token = token.lower()

            # Now decide based on BIO label
            if label == 1:  # B - begin new phrase
                if current_phrase_tokens:
                    phrase = " ".join(current_phrase_tokens)
                    if len(phrase) > 2 and phrase not in STOPWORDS:  # filter short trash & stopwords
                        phrases.append(phrase)
                current_phrase_tokens = [cleaned_token]

            elif label == 2:  # I - continue phrase
                if current_phrase_tokens:  # only if we are inside a phrase
                    current_phrase_tokens.append(cleaned_token)
                else:
                    # Orphan I-tag (sometimes happens due to model error), treat as B
                    current_phrase_tokens = [cleaned_token]

            else:  # O - outside
                if current_phrase_tokens:
                    phrase = " ".join(current_phrase_tokens)
                    if len(phrase) > 2 and phrase not in STOPWORDS:
                        phrases.append(phrase)
                    current_phrase_tokens = []

    # Don't forget the last phrase
    if current_phrase_tokens:
        phrase = " ".join(current_phrase_tokens)
        if len(phrase) > 2 and phrase not in STOPWORDS:
            phrases.append(phrase)

    # Optional: post-processing to remove very common garbage still slipping through
    phrases = [p for p in phrases if len(p) <= 30]  # remove ridiculously long broken phrases
    phrases = list(dict.fromkeys(phrases))  # dedupe while preserving order

    return phrases



# ---------------------
# Attention Heatmap
# ---------------------


def get_attention_weights(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        x = bert_emb(input_ids)
        x = projection(x)
        enc_out, attn_weights = encoder(x, mask=attention_mask)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    return attn_weights, tokens

def plot_attention_heatmap(attn, tokens, head=1, show_values=False):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="viridis",
        annot=show_values,
        fmt=".2f",
        cbar=True,
        ax=ax
    )
    ax.set_xlabel("Key Tokens")
    ax.set_ylabel("Query Tokens")
    ax.set_title(f"Attention Heatmap - Head {head}")
    st.pyplot(fig)
    plt.close(fig)


def get_attention_weights_cached(text):
    if "attn_cache" not in st.session_state or st.session_state["last_text"] != text:
        attn_weights, tokens = get_attention_weights(text)
        st.session_state["attn_cache"] = attn_weights
        st.session_state["tokens_cache"] = tokens
        st.session_state["last_text"] = text
    return st.session_state["attn_cache"], st.session_state["tokens_cache"]



# ###########
# # predict_ner_tags
# ###########

# def predict_ner_tags(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
#     input_ids = inputs["input_ids"].to(device)
#     attention_mask = inputs["attention_mask"].to(device)

#     with torch.no_grad():
#         x = bert_emb(input_ids)
#         x = projection(x)
#         enc_out, _ = encoder(x, mask=attention_mask)
#         logits = classifier(enc_out)
#         preds = torch.argmax(logits, dim=-1)[0].cpu().tolist()

#     tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
#     return tokens, preds

# def format_ner_html(tokens, preds):
#     tag2color = {
#         0: "transparent",       # O - no background
#         1: "#28a745",   # Green (Bootstrap success)
#         2: "#7dd77d"    # Light green
#     }
#     tag2textcolor = {
#         0: "white",
#         1: "black",             # white text on blue bg
#         2: "white"              # white text on lighter blue bg
#     }
#     html_text = ""
#     for tok, pred in zip(tokens, preds):
#         if tok in ("[CLS]", "[SEP]"):
#             continue
#         tok_clean = tok.replace("##", "")
#         bg_color = tag2color.get(pred, "transparent")
#         text_color = tag2textcolor.get(pred, "black")
#         html_text += f"<span style='background-color:{bg_color}; color:{text_color}; padding:2px 6px; margin:2px; border-radius:4px;'>{tok_clean} </span>"
#     return html_text


# ────────────────────── FIXED & ROBUST PREDICTION + MERGING ──────────────────────
def predict_ner_tags(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    
    with torch.no_grad():
        x = bert_emb(inputs["input_ids"])
        x = projection(x)
        enc_out, _ = encoder(x, mask=inputs["attention_mask"])
        logits = classifier(enc_out)
        preds = torch.argmax(logits, dim=-1)[0].cpu().tolist()

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return tokens, preds


def merge_subwords_and_bio(tokens, preds):
    """
    Converts subword tokens + subword-level BIO tags → full words + correct word-level BIO
    """
    words = []
    final_tags = []
    current_word = ""
    current_tag = None

    tag_map = {0: "O", 1: "B", 2: "I"}
    
    for tok, pred in zip(tokens, preds):

        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            continue

        raw_tok = tok.replace("##", "")

        if tok.startswith("##"):
            # Subword continuation
            current_word += raw_tok

            if final_tags:
                prev = final_tags[-1]

                if "-" in prev:
                    # Safe split
                    parts = prev.split("-")
                    if len(parts) >= 2:
                        final_tags[-1] = "I-" + parts[1]
                    else:
                        final_tags[-1] = "I"
                else:
                    final_tags[-1] = "I"

        else:
            # New word
            if current_word:
                if current_tag in ("B", "I"):
                    final_tags.append("B-ENT" if current_tag == "B" else "I-ENT")
                else:
                    final_tags.append("O")

                words.append(current_word)

            current_word = raw_tok
            current_tag = tag_map[pred]

    # Last word
    if current_word:
        if current_tag in ("B", "I"):
            final_tags.append("B-ENT" if current_tag == "B" else "I-ENT")
        else:
            final_tags.append("O")
        words.append(current_word)

    # Fix BIO transitions
    for i in range(1, len(final_tags)):
        if final_tags[i].startswith("B-") and final_tags[i - 1].startswith("I-"):
            final_tags[i] = "I-" + final_tags[i].split("-")[1]

    return words, final_tags



def render_ner_html(words, tags):
    html = ""
    for word, tag in zip(words, tags):
        if tag == "O":
            html += f"<span style='padding:2px 4px; margin:2px;'>{word}</span>"
        else:
            color = "#ff6b6b"   # Coral red
            if "LOC" in tag or "GPE" in tag or "ENT" in tag:
                color = "#4ecdc4"  # Turquoise
            elif "PER" in tag:
                color = "#f7b731"  # Orange
            elif "ORG" in tag:
                color = "#a55eea"  # Purple

            html += f"<span style='background-color:{color}; color:white; padding:6px 10px; margin:3px; border-radius:8px; font-weight:600;'>{word}</span>"
    return html



