import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertModel
import torch
import torch.nn as nn
from encoder_layer import Encoder_block
import pandas as pd
from nlp_functions import extract_keyphrases, get_attention_weights, plot_attention_heatmap, predict_ner_tags, render_ner_html, merge_subwords_and_bio
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------
# Load Model Components
# ---------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
bert_emb = DistilBertModel.from_pretrained("distilbert-base-uncased").get_input_embeddings().to(device)

d_model = 512
num_heads = 8
d_ff = 2048

projection = nn.Linear(768, 512)
encoder = Encoder_block(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
classifier = nn.Linear(512, 3)

ckpt = torch.load("restored_model.pt", map_location=device)

projection.load_state_dict(ckpt["projection_state_dict"])
encoder.load_state_dict(ckpt["encoder_state_dict"])
classifier.load_state_dict(ckpt["classifier_state_dict"])

projection.to(device)
encoder.to(device)
classifier.to(device)

encoder.eval()
classifier.eval()

# ---------------------
# Streamlit Frontend
# ---------------------

feature = st.sidebar.radio(options=["Extract Keywords", "Attention Heatmap Viewer", "NER Demo"], label="Features")

st.sidebar.markdown("")

st.sidebar.image("encoder2.png", width=200)

if feature == "Extract Keywords":
    # st.title("🔍 Keyword Extraction using Custom Transformer Encoder")
    st.markdown("<h1 style='color:green; text-align: center; margin-top: -40px;'>Keyword Extraction using Custom Transformer Encoder</h1>", unsafe_allow_html=True)
    

    # st.write("Enter text below and get extracted keyphrases:")
    st.markdown("<h6 style='color:violet; text-align: center;'>Enter text below and get extracted keyphrases:</h6>", unsafe_allow_html=True)

    

    user_text = st.text_area("Input Text", height=100)

    # ---------------------
    # Keyword Extraction
    # ---------------------

    if st.button("Extract Keywords"):
        if not user_text.strip():
            st.warning("Please enter some text!")
        else:
            with st.spinner("Extracting keywords..."):
                keyphrases = extract_keyphrases(user_text)

            # st.subheader("✨ Extracted Keyphrases")
            st.markdown("<h4 style='color:blue;'>✨ Extracted Keyphrases</h4>", unsafe_allow_html=True)

            if len(keyphrases) == 0:
                st.info("No meaningful keyphrases detected.")
            else:
                # for kp in keyphrases:
                #     st.success(kp)
            
                html = "<div style='display: flex; gap: 8px; flex-wrap: wrap;'>"
                for kp in keyphrases:
                    html += f"<div style='background-color:#d4edda; color:#155724; padding:8px 12px; border-radius:12px; font-weight:600;'>" \
                            f"{kp}</div>"
                html += "</div>"

                st.markdown(html, unsafe_allow_html=True)

# feature = st.sidebar.selectbox("Choose Feature", ["Keyword Extraction", "Attention Heatmap Viewer"])

# if feature == "Attention Heatmap Viewer":

#     # st.title("Attention Heatmap Viewer")
#     st.markdown("<h2 style='color:green; text-align: center;'>Attention Heatmap Viewer</h2>", unsafe_allow_html=True)
    
#     sentence = st.text_area("Enter a sentence:", value="The quick brown fox jumps over the lazy dog")

    

#     if sentence.strip() != "":
#         if "last_text" not in st.session_state or st.session_state["last_text"] != sentence:
#             st.session_state["last_text"] = sentence
#             st.session_state["attn_cache"], st.session_state["tokens_cache"] = get_attention_weights(sentence)
        
#         attn_weights = st.session_state["attn_cache"]
#         tokens = st.session_state["tokens_cache"]

#         num_heads1 = attn_weights.shape[1]
#         head_display_options = list(range(1, num_heads1 + 1)) 
#         head_display  = st.selectbox(
#             "Select an attention head to visualize:",
#             options=head_display_options,
#             index=0
#         )
#         head = head_display - 1

#         attn = attn_weights[0, head].cpu().numpy()
#         unique_tokens = [f"{tok}_{i}" for i, tok in enumerate(tokens)]

#         plot_attention_heatmap(attn, tokens, head)

#         st.markdown(f"**Raw Attention Scores - Head {head}:**")
#         df_attn = pd.DataFrame(attn, columns=unique_tokens, index=unique_tokens)
#         st.dataframe(df_attn.style.format("{:.7f}"))



# if feature == "Attention Heatmap Viewer":

#     # st.title("Attention Heatmap Viewer")
#     st.markdown("<h2 style='color:green; text-align: center; margin-top: -40px;'>Attention Heatmap Viewer</h2>", unsafe_allow_html=True)
    
#     if "attn_cache" not in st.session_state:
#         st.session_state.attn_cache = None
#     if "tokens_cache" not in st.session_state:
#         st.session_state.tokens_cache = None
#     if "last_text" not in st.session_state:
#         st.session_state.last_text = ""

#     col1, col2 = st.columns([4, 1])

#     with col1:
#         with st.form("input_form"):
#             sentence = st.text_area("Enter a sentence:", value=st.session_state.last_text)
#             submitted = st.form_submit_button("See HeatMap")

#     if submitted and sentence.strip() != "":
#         st.session_state.last_text = sentence
#         st.session_state.attn_cache, st.session_state.tokens_cache = get_attention_weights(sentence)

#     if st.session_state.attn_cache is not None:
#         attn_weights = st.session_state.attn_cache
#         tokens = st.session_state.tokens_cache

#         num_heads = attn_weights.shape[1]
#         head_display_options = list(range(1, num_heads + 1))

#     with col2:
#         head_display = st.selectbox(
#             "Select an attention head to visualize:",
#             options=head_display_options,
#             index=0
#         )
#         head = head_display - 1

#         attn = attn_weights[0, head].cpu().numpy()
#         unique_tokens = [f"{tok}_{i}" for i, tok in enumerate(tokens)]

#         plot_attention_heatmap(attn, tokens, head_display)
#         st.markdown(f"**Raw Attention Scores - Head {head_display}:**")
#         import pandas as pd
#         df_attn = pd.DataFrame(attn, columns=unique_tokens, index=unique_tokens)
#         st.dataframe(df_attn.style.format("{:.7f}"))


if feature == "Attention Heatmap Viewer":
    st.markdown("<h1 style='color:green; text-align: center; margin-top: -40px; margin-bottom: 10px;'>Attention Heatmap Viewer</h1>", unsafe_allow_html=True)
    
    # Initialize session state
    if "attn_cache" not in st.session_state:
        st.session_state.attn_cache = None
    if "tokens_cache" not in st.session_state:
        st.session_state.tokens_cache = None
    if "last_text" not in st.session_state:
        st.session_state.last_text = ""

    col1, col2 = st.columns([4, 1])

    with col1:
        with st.form("input_form"):
            sentence = st.text_area("Enter a sentence:", value=st.session_state.last_text, height=100)
            submitted = st.form_submit_button("See HeatMap")

        # Process input when submitted
        if submitted and sentence.strip():
            with st.spinner("Computing attention weights..."):
                st.session_state.last_text = sentence
                st.session_state.attn_cache, st.session_state.tokens_cache = get_attention_weights(sentence)
            st.success("Attention weights computed!")

    # Only show visualization if we have attention data
    if st.session_state.attn_cache is not None:
        attn_weights = st.session_state.attn_cache
        tokens = st.session_state.tokens_cache

        num_heads = attn_weights.shape[1]
        head_options = list(range(1, num_heads + 1))  # Define here so it's always available when data exists

        with col2:
            head_display = st.selectbox(
                "Select an attention head to visualize:",
                options=head_options,
                index=0,
                key="head_selector"  # Prevents conflicts
            )
        
        head_idx = head_display - 1
        attn = attn_weights[0, head_idx].cpu().numpy()

        # Make tokens unique for display (especially important for repeated tokens)
        unique_tokens = [f"{tok}_{i}" for i, tok in enumerate(tokens)]
        display_tokens = tokens  # For clean visualization

        # Plot heatmap
        plot_attention_heatmap(attn, display_tokens, head_display)

        # Show raw attention matrix
        st.markdown(f"**Raw Attention Scores - Head {head_display}:**")
        import pandas as pd
        df_attn = pd.DataFrame(attn, columns=unique_tokens, index=unique_tokens)
        st.dataframe(df_attn.style.format("{:.6f}").background_gradient(cmap='viridis'))

    else:
        # Show a placeholder when no data yet
        with col2:
            st.selectbox(
                "Select an attention head to visualize:",
                options=[1],  # dummy
                disabled=True
            )
        st.info("👈 Enter a sentence and click 'See HeatMap' to visualize attention!")


# if feature == "NER Demo":

#     # st.title("NER Demo with Custom Transformer Encoder")
#     st.markdown("<h2 style='color:green; text-align: center;'>NER Demo with Custom Transformer Encoder</h2>", unsafe_allow_html=True)


#     user_text = st.text_area("Enter your sentence:")

#     if st.button("Predict NER Tags"):
#         if not user_text.strip():
#             st.warning("Please enter some text!")
#         else:
#             tokens, preds = predict_ner_tags(user_text)
#             ner_html = format_ner_html(tokens, preds)
#             # st.markdown("### Predicted Named Entities")
#             st.markdown("<h4 style='color:yellow; text-align: center;'>Predicted Named Entities</h4>", unsafe_allow_html=True)
#             st.markdown(ner_html, unsafe_allow_html=True)

#             # Optionally show token-tag pairs
#             tag_map = {0: "O", 1: "B", 2: "I"}
#             tag_pairs = [(t.replace("##", ""), tag_map[p]) for t, p in zip(tokens, preds) if t not in ("[CLS]", "[SEP]")]
#             st.table(tag_pairs)


if feature == "NER Demo":
    st.markdown("<h1 style='color:green; text-align: center; margin-top: -40px;'>NER Demo with Custom Transformer Encoder</h1>", unsafe_allow_html=True)
    # st.markdown("Enter a sentence below and see **word-level** named entity recognition with clean highlighting.")

    st.markdown("<h6 style='color:yellow; text-align: center; font-weight: 400;font-size: 14px;'>Enter a sentence below and see **word-level** named entity recognition with clean highlighting.</h6>", unsafe_allow_html=True)


    user_text = st.text_area("Enter your sentence:", height=120, placeholder="Example: Apple is opening its first store in Delhi, India next month.")

    if st.button("Predict Named Entities", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("Please enter some text!")
        else:
            with st.spinner("Running your custom NER model..."):
                tokens, preds = predict_ner_tags(user_text)
                words, tags = merge_subwords_and_bio(tokens, preds)
                ner_html = render_ner_html(words, tags)

            # Beautiful highlighted output
            st.markdown("<h4 style='color:gold; text-align: center;'>Predicted Entities</h4>", unsafe_allow_html=True)

            # FIXED: Fully responsive, wraps naturally, never goes off-screen
            st.markdown(f"""
            <div style="
                font-size: 19px;
                line-height: 2.4;
                padding: 22px;
                background: #1e1e1e;
                border-radius: 16px;
                border-left: 6px solid #ffd700;
                overflow-wrap: break-word;
                word-wrap: break-word;
                white-space: normal;
                text-align: justify;
            ">
                {ner_html}
            </div>
            """, unsafe_allow_html=True)

            # Clean table
            st.markdown("#### Token → NER Tag")
            df = pd.DataFrame({
                "Word": words,
                "BIO Tag": tags,
                "Entity": [tag[2:] if tag.startswith(("B-", "I-")) else "O" for tag in tags],
                "Full Label": [tag if tag != "O" else "Outside" for tag in tags]
            })
            st.dataframe(df, use_container_width=True, hide_index=True)