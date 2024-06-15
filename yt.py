import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import BartTokenizer, BartForConditionalGeneration
import logging
import re


logging.basicConfig(level=logging.INFO)


tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')


st.title("YouTube Video Transcript Summarizer")


video_link = st.text_input("Enter YouTube video link:")

@st.cache_data
def fetch_transcript(video_id):
    return YouTubeTranscriptApi.get_transcript(video_id)

@st.cache_resource
def summarize_text(text):
    max_input_length = 1024  
    logging.info(f"Summarizing text with length: {len(text)}")
    input_tensor = tokenizer.encode(text[:max_input_length], return_tensors="pt", max_length=512, truncation=True)
    outputs_tensor = model.generate(input_tensor, max_length=160, min_length=120, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs_tensor[0], skip_special_tokens=True)
    logging.info("Summary generated successfully")
    return summary

def extract_video_id(link):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", link)
    return match.group(1) if match else None

if video_link:
    try:
      
        video_id = extract_video_id(video_link)
        if not video_id:
            st.error("Invalid YouTube video link")
        else:
            
            transcript = fetch_transcript(video_id)
            transcript_text = " ".join([entry['text'] for entry in transcript])
            
            
            st.subheader("Original Transcript")
            st.write(transcript_text)
            
            
            st.download_button(
                label="Download Original Transcript",
                data=transcript_text,
                file_name='original_transcript.txt',
                mime='text/plain'
            )
            
            
            summary = summarize_text(transcript_text)
            

            st.subheader("Summarized Transcript")
            st.write(summary)
            
          
            st.download_button(
                label="Download Summary",
                data=summary,
                file_name='summary.txt',
                mime='text/plain'
            )
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        st.error(f"An error occurred: {e}")
