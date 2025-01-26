import streamlit as st
import pdfplumber
from openai import OpenAI
import tempfile
import os
import json
from typing import Optional, List, Dict
import numpy as np

def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks of approximately equal size."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        # Find the end of the chunk
        end = start + chunk_size
        
        # If this is not the last chunk, try to find a natural breaking point
        if end < len(text):
            # Look for the last period or newline in the overlap region
            last_period = text.rfind('.', end - overlap, end)
            last_newline = text.rfind('\n', end - overlap, end)
            break_point = max(last_period, last_newline)
            
            if break_point != -1:
                end = break_point + 1
        else:
            end = len(text)
        
        chunks.append(text[start:end])
        start = end - overlap if end < len(text) else end
    
    return chunks

def extract_text_from_pdf(pdf_file) -> Optional[str]:
    """Extract text from uploaded PDF file."""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n\n"
            return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

def process_chunk_with_openai(client, chunk: str, prompt: str) -> Dict:
    """Process a single chunk with OpenAI API."""
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that analyzes PDF documents. "
             "Extract all relevant information and return it in a detailed JSON format."},
            {"role": "user", "content": f"Text chunk to analyze:\n{chunk}\n\nInstruction: {prompt}\n"
             "Return the information in a detailed JSON format, capturing all relevant details from the text."}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        # Parse the response as JSON
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": str(e), "chunk": chunk[:100] + "..."}

def merge_json_results(results: List[Dict]) -> Dict:
    """Merge multiple JSON results into a single coherent structure."""
    merged = {}
    
    for result in results:
        for key, value in result.items():
            if key not in merged:
                merged[key] = value
            elif isinstance(value, list):
                if isinstance(merged[key], list):
                    merged[key].extend(value)
                else:
                    merged[key] = [merged[key], *value]
            elif isinstance(value, dict):
                if isinstance(merged[key], dict):
                    merged[key].update(value)
                else:
                    merged[key] = value
            else:
                if isinstance(merged[key], list):
                    merged[key].append(value)
                else:
                    merged[key] = [merged[key], value]
    
    return merged

def main():
    st.title("PDF Analysis with Chunking")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter your OpenAI API key", type="password")
        chunk_size = st.number_input("Chunk size (words)", value=2000, min_value=500, max_value=5000)
        overlap_size = st.number_input("Overlap size (words)", value=200, min_value=50, max_value=500)
        
        st.markdown("""
        ### How to use:
        1. Enter your OpenAI API key
        2. Upload a PDF file
        3. Enter your analysis prompt
        4. View the combined JSON results
        """)

    # Initialize OpenAI client if API key is provided
    client = None
    if api_key:
        client = OpenAI(api_key=api_key)

    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Extract text
        extracted_text = extract_text_from_pdf(tmp_path)
        os.unlink(tmp_path)  # Delete temporary file
        
        if extracted_text:
            # Show extracted text in expandable section
            with st.expander("View Extracted Text"):
                st.text_area("PDF Content", extracted_text, height=300)
            
            # Query input
            prompt = st.text_area("Enter your analysis prompt:",
                               placeholder="Example: Extract all information into a structured JSON format")
            
            # Process chunks
            if st.button("Analyze Document") and prompt:
                if client:
                    chunks = chunk_text(extracted_text, chunk_size, overlap_size)
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    st.write(f"Processing {len(chunks)} chunks...")
                    
                    # Process each chunk
                    results = []
                    for i, chunk in enumerate(chunks):
                        with st.spinner(f"Processing chunk {i+1}/{len(chunks)}..."):
                            chunk_result = process_chunk_with_openai(client, chunk, prompt)
                            results.append(chunk_result)
                            progress_bar.progress((i + 1) / len(chunks))
                    
                    # Merge results
                    final_result = merge_json_results(results)
                    
                    # Display results
                    st.markdown("### Combined Analysis Results")
                    st.json(final_result)
                    
                    # Download button for JSON
                    st.download_button(
                        label="Download JSON Results",
                        data=json.dumps(final_result, indent=2),
                        file_name="analysis_results.json",
                        mime="application/json"
                    )
                else:
                    st.error("Please enter your OpenAI API key in the sidebar first.")

if __name__ == "__main__":
    main()