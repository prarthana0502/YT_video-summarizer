import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import BartTokenizer, BartForConditionalGeneration

# Initialize BART tokenizer and model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Streamlit app
st.title("YouTube Video Transcript Summarizer")

# Input: YouTube video link
video_link = st.text_input("Enter YouTube video link:")

def summarize_text(text):
    # Limit the input length to avoid memory issues
    max_input_length = 1024  # Adjust as necessary
    input_tensor = tokenizer.encode(text[:max_input_length], return_tensors="pt", max_length=512, truncation=True)
    outputs_tensor = model.generate(input_tensor, max_length=160, min_length=120, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs_tensor[0], skip_special_tokens=True)
    return summary

if video_link:
    try:
        # Extract unique video ID from the link
        unique_id = video_link.split("/")[-1].split("?")[0]
        
        # Fetch video transcript
        sub = YouTubeTranscriptApi.get_transcript(unique_id)
        subtitle = " ".join([x['text'] for x in sub])
        
        # Display the original transcript
        st.subheader("Original Transcript")
        st.write(subtitle)
        
        # Provide download button for the original transcript
        st.download_button(
            label="Download Original Transcript",
            data=subtitle,
            file_name='original_transcript.txt',
            mime='text/plain'
        )
        
        # Summarize the transcript
        summary = summarize_text(subtitle)
        
        # Display the summary
        st.subheader("Summarized Transcript")
        st.write(summary)
        
        # Provide download button for the summary
        st.download_button(
            label="Download Summary",
            data=summary,
            file_name='summary.txt',
            mime='text/plain'
        )
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
