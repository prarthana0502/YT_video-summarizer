import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import BartTokenizer, BartForConditionalGeneration
import logging
import re
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize BART tokenizer and model
@st.cache_resource
def load_model():
    logging.info("Loading BART tokenizer and model")
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit app
st.title("YouTube Video Transcript Summarizer")

# Input: YouTube video link
video_link = st.text_input("Enter YouTube video link:")

@st.cache_data
def fetch_transcript(video_id):
    return YouTubeTranscriptApi.get_transcript(video_id)

@st.cache_data
def summarize_text(text):
    max_input_length = 1024  # Adjust as necessary
    logging.info(f"Summarizing text with length: {len(text)}")
    input_tensor = tokenizer.encode(text[:max_input_length], return_tensors="pt", max_length=512, truncation=True)
    outputs_tensor = model.generate(input_tensor, max_length=160, min_length=120, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs_tensor[0], skip_special_tokens=True)
    logging.info("Summary generated successfully")
    return summary

def extract_video_id(link):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", link)
    return match.group(1) if match else None

def display_text_and_download_button(header, text, filename):
    st.subheader(header)
    st.write(text)
    st.download_button(
        label=f"Download {header}",
        data=text,
        file_name=filename,
        mime='text/plain'
    )

if video_link:
    try:
        # Extract unique video ID from the link
        video_id = extract_video_id(video_link)
        if not video_id:
            st.error("Invalid YouTube video link")
        else:
            # Fetch video transcript
            transcript = fetch_transcript(video_id)
            transcript_text = " ".join([entry['text'] for entry in transcript])
            
            # Display and download original transcript
            display_text_and_download_button("Original Transcript", transcript_text, 'original_transcript.txt')
            
            # Summarize the transcript
            summary = summarize_text(transcript_text)
            
            # Display and download summary
            display_text_and_download_button("Summarized Transcript", summary, 'summary.txt')
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        st.error(f"An error occurred: {e}\n\n{traceback.format_exc()}")
