import streamlit as st
import os
import numpy as np
import faiss
import json
from mistralai import Mistral, UserMessage

# Page Configuration
st.set_page_config(
    page_title="UDST Policy Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Design
st.markdown("""
    <style>
        /* General App Styling */
        .stApp {
            max-width: 1100px;
            margin: 0 auto;
            font-family: 'Arial', sans-serif;
        }
        .main-container {
            display: flex;
            gap: 20px;
        }
        .left-panel {
            width: 30%;
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }
        .right-panel {
            width: 70%;
            padding: 20px;
        }
        .answer-box {
            background: #f4f4f4;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0px 3px 8px rgba(0, 0, 0, 0.1);
        }
        .stButton>button {
            width: 100%;
            padding: 10px;
            border-radius: 8px;
            font-weight: bold;
            background-color: #007bff;
            color: white;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Information
with st.sidebar:
    st.header("About UDST Policy Assistant")
    st.write("This assistant helps answer questions about UDST policies using Retrieval-Augmented Generation (RAG).")
    
    api_key = "tlcYsUNSS1iVHZ6lWnUw8KKW2f8AoVJf"

    if not os.path.exists("indexes") or len(os.listdir("indexes")) == 0:
        st.error("Policy data not found. Please run the preprocess.py script first.")
    else:
        st.success(f"Found data for {len(os.listdir('indexes'))} policies")

# Policy Data
POLICY_URLS = {
    "Sport and Wellness Facilities": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/sport-and-wellness-facilities-and",
    "Credit Hour Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/credit-hour-policy",
    "Final Grade Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/final-grade-policy",
    "Student Appeals Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-appeals-policy",
}

# Function to Get Available Policies
def get_available_policies():
    if not os.path.exists("indexes"):
        return []
    
    available = []
    for policy, url in POLICY_URLS.items():
        safe_name = policy.lower().replace(" ", "_")
        if os.path.exists(f"indexes/{safe_name}.index") and os.path.exists(f"chunks/{safe_name}_chunks.json"):
            available.append(policy)
    
    return available

# Function to Query Mistral
def query_mistral(prompt, api_key):
    client = Mistral(api_key=api_key)
    messages = [UserMessage(content=prompt)]
    try:
        chat_response = client.chat.complete(
            model="mistral-large-latest",
            messages=messages,
        )
        return chat_response.choices[0].message.content
    except Exception as e:
        return f"Error: Unable to generate response. {str(e)}"

# Main UI Layout
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

# Left Panel - Policy Selection
st.markdown("<div class='left-panel'>", unsafe_allow_html=True)
st.subheader("Select a Policy")
available_policies = get_available_policies()
selected_policy = st.selectbox("Choose a policy:", available_policies)

st.markdown("</div>", unsafe_allow_html=True)

# Right Panel - Q&A Section
st.markdown("<div class='right-panel'>", unsafe_allow_html=True)
st.subheader("Ask a Question")

query = st.text_input("Enter your question:")

if query and selected_policy:
    with st.spinner("Fetching the best answer..."):
        try:
            # Load policy chunks
            safe_name = selected_policy.lower().replace(" ", "_")
            chunks_path = f"chunks/{safe_name}_chunks.json"
            index_path = f"indexes/{safe_name}.index"

            if not os.path.exists(chunks_path) or not os.path.exists(index_path):
                st.error(f"Data for {selected_policy} not found.")
            else:
                with open(chunks_path, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                index = faiss.read_index(index_path)

                # Generate query embedding
                query_embedding_data = np.random.rand(1, 512)  # Dummy embedding for testing
                query_embedding = np.array(query_embedding_data)

                # Retrieve relevant chunks
                k = min(3, len(chunks))
                D, I = index.search(query_embedding, k)
                retrieved_chunks = [chunks[i] for i in I.tolist()[0]]

                # Create prompt
                context = "\n".join(retrieved_chunks)
                prompt = f"""
                Context:
                ---------------------
                {context}
                ---------------------
                
                Answer the following question based ONLY on the context provided:
                {query}
                """

                # Get response
                response = query_mistral(prompt, api_key)

                # Display answer
                st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
                st.subheader("Answer:")
                st.write(response)
                st.markdown("</div>", unsafe_allow_html=True)

                # Show Sources
                with st.expander("Source Context"):
                    for i, chunk in enumerate(retrieved_chunks):
                        st.markdown(f"**Source {i+1}:**")
                        st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("© 2025 UDST Policy Assistant - Modern UI Edition")
