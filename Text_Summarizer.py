
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
from PyPDF2 import PdfReader
from docx import Document

# Initialize the model
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# UI Design Improvements
st.set_page_config(page_title="AI Text Summarizer", page_icon="📄", layout="wide")

# Header Section
st.markdown("""
    <style>
        body {
            background-color: #f7f7f7;
            font-family: 'Poppins', sans-serif;
        }
        .header {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 2.8rem;
            font-weight: bold;
            margin: 0;
        }
        .header p {
            font-size: 1.2rem;
            color: #7f8c8d;
        }
    </style>
    <div class="header">
        <h1>AI-Powered Text Summarizer</h1>
        <p>Summarize Text, PDFs, and DOCX files effortlessly.</p>
    </div>
""", unsafe_allow_html=True)

# Input Section
st.sidebar.header("Choose Input Type")
input_type = st.sidebar.radio("Select the input format:", ("Text", "PDF", "DOCX"))

input_data = None

if input_type == "Text":
    input_data = st.text_area("Enter Text to Summarize:", height=200)
elif input_type == "PDF":
    uploaded_pdf = st.file_uploader("Upload a PDF File:", type=["pdf"])
    if uploaded_pdf:
        reader = PdfReader(uploaded_pdf)
        input_data = " ".join([page.extract_text() for page in reader.pages])
        st.success("PDF content extracted!")
elif input_type == "DOCX":
    uploaded_docx = st.file_uploader("Upload a DOCX File:", type=["docx"])
    if uploaded_docx:
        doc = Document(uploaded_docx)
        input_data = " ".join([para.text for para in doc.paragraphs])
        st.success("DOCX content extracted!")

# Summarize Button
if st.button("Summarize") and input_data:
    with st.spinner("Summarizing... Please wait."):
        inputs = tokenizer.encode("summarize: " + input_data, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.text_area("Summary:", value=summary, height=200)

        # Copy Button
        st.markdown("""
            <style>
                .copy-btn {
                    background-color: #3498db;
                    color: white;
                    border: none;
                    padding: 8px 15px;
                    font-size: 14px;
                    border-radius: 5px;
                    cursor: pointer;
                }
                .copy-btn:hover {
                    background-color: #2980b9;
                }
            </style>
            <button class="copy-btn" onclick="navigator.clipboard.writeText('{}')">Copy Summary</button>
        """.format(summary), unsafe_allow_html=True)
else:
    st.info("Please provide input data to summarize.")

# Footer
st.markdown("""
    <style>
        .footer {
            text-align: center;
            margin-top: 30px;
            color: #95a5a6;
        }
        .footer a {
            color: #3498db;
            text-decoration: none;
        }
        .footer a:hover {
            text-decoration: underline;
        }
    </style>
    <div class="footer">
        <p>A project by <a href="https://www.linkedin.com/in/saiharshithh" target="_blank">Sai Harshith</a>.</p>
    </div>
""", unsafe_allow_html=True)
