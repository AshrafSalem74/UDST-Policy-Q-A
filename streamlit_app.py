import streamlit as st
import os
import numpy as np
import faiss
import json
from mistralai import Mistral, UserMessage

# Page configuration
st.set_page_config(
    page_title="UDST Policy AI Assistant",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern appearance
st.markdown("""
<style>
    /* Main content styling */
    .stApp {
        background-color: black;
    }
    
    /* Header styling */
    .header {
        background-color: #00205B;
        padding: 2rem;
        border-radius: 0 0 1rem 1rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Policy selector styling */
    .policy-selector {
        background-color: black;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
    
    /* Answer card styling */
    .answer-card {
        background-color: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0,32,91,0.1);
        margin-top: 1.5rem;
        border-left: 4px solid #00205B;
    }
    
    /* Source card styling */
    .source-card {
        background-color: black;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #00205B;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 2rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Mistral API Configuration
api_key = "tlcYsUNSS1iVHZ6lWnUw8KKW2f8AoVJf"

# Policy URLs list
POLICY_URLS = {
    "Sport and Wellness Facilities": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/sport-and-wellness-facilities-and",
    "Credit Hour Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/credit-hour-policy",
    "Final Grade Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/final-grade-policy",
    "Student Appeals Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-appeals-policy",
    "Student Attendance Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-attendance-policy",
    "Student Counselling Services": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/student-counselling-services-policy",
    "Scholarship and Financial Assistance Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/scholarship-and-financial-assistance",
    "Transfer Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/transfer-policy",
    "Academic Schedule Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/academic-schedule-policy",
    "Registarion Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/registration-policy"
}

# Function to get safe filename from policy name
def get_safe_filename(policy_name):
    return policy_name.lower().replace(" ", "_")

# Function to get text embeddings
def get_text_embedding(list_txt_chunks, api_key):
    client = Mistral(api_key=api_key)
    embeddings_batch_response = client.embeddings.create(
        model="mistral-embed",
        inputs=list_txt_chunks
    )
    return embeddings_batch_response.data

# Function to query Mistral
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
        try:
            chat_response = client.chat.complete(
                model="mistral-small-latest",
                messages=messages,
            )
            return chat_response.choices[0].message.content + "\n\n(Note: Used fallback model)"
        except:
            return f"Error: Unable to generate response. {str(e)}"

# Get available policies
def get_available_policies():
    if not os.path.exists("indexes"):
        return []
    
    available = []
    for policy, url in POLICY_URLS.items():
        safe_name = get_safe_filename(policy)
        if os.path.exists(f"indexes/{safe_name}.index") and os.path.exists(f"chunks/{safe_name}_chunks.json"):
            available.append(policy)
    return available

# Main header
st.markdown("""
<div class="header">
    <h1 style="color: white; margin: 0;">📚 UDST Policy Assistant</h1>
    <p style="color: #ffffffcc; margin: 0.5rem 0 0;">Your AI-powered guide to university policies</p>
</div>
""", unsafe_allow_html=True)

# Sidebar content
with st.sidebar:
    st.image("https://www.udst.edu.qa/themes/custom/cnaq/logo-white.svg", width=200)
    st.markdown("---")
    st.markdown("### How to Use")
    st.markdown("""
    1. Select a policy from the dropdown
    2. Ask your question in natural language
    3. Review the AI-powered answer
    4. Explore source references
    """)
    
    st.markdown("---")
    st.markdown("### Supported Policies")
    for policy in get_available_policies():
        st.markdown(f"- {policy}")
    
    st.markdown("---")
    st.caption("Version 1.0 | Powered by Mistral AI")

# Main content area
st.markdown("""
<div class="policy-selector">
    <h3 style="margin-top: 0;">🔍 Start Your Policy Inquiry</h3>
""", unsafe_allow_html=True)

# Policy selection and query input
col1, col2 = st.columns([1, 2])
with col1:
    selected_policy = st.selectbox(
        "Select Policy Document",
        get_available_policies(),
        index=0,
        help="Choose which policy document you want to query"
    )

with col2:
    query = st.text_input(
        "Enter Your Question",
        placeholder="e.g. What are the attendance requirements?",
        help="Ask your question in natural language"
    )

st.markdown("</div>", unsafe_allow_html=True)

# Processing and response
if query and selected_policy:
    with st.spinner("🔍 Analyzing policy documents..."):
        try:
            safe_name = get_safe_filename(selected_policy)
            chunks_path = f"chunks/{safe_name}_chunks.json"
            index_path = f"indexes/{safe_name}.index"

            # Load chunks and index
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            
            index = faiss.read_index(index_path)
            
            # Generate query embedding
            query_embedding_data = get_text_embedding([query], api_key)
            query_embedding = np.array([query_embedding_data[0].embedding])
            
            # Search index
            k = min(3, len(chunks))
            D, I = index.search(query_embedding, k)
            retrieved_chunks = [chunks[i] for i in I.tolist()[0]]
            context = "\n".join(retrieved_chunks)
            
            # Create prompt
            prompt = f"""
            Context information about {selected_policy} is below:
            ---------------------
            {context}
            ---------------------
            Based ONLY on the context information provided, answer this query:
            Query: {query}
            Answer:
            """
            
            # Get response
            response = query_mistral(prompt, api_key)
            
            # Display response
            st.markdown("""
            <div class="answer-card">
                <h3 style="margin-top: 0; color: #00205B;">📝 AI Analysis</h3>
            """, unsafe_allow_html=True)
            st.markdown(response)
            
            # Source sections
            with st.expander("🔖 View Supporting Sources", expanded=True):
                for i, chunk in enumerate(retrieved_chunks):
                    st.markdown(f"""
                    <div class="source-card">
                        <h4>Source Excerpt #{i+1}</h4>
                        <p>{chunk[:300] + "..." if len(chunk) > 300 else chunk}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ Error processing request: {str(e)}")

elif not selected_policy:
    st.info("ℹ️ Please select a policy document to continue")
else:
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #666;">
        <h3>💡 How to Get Started</h3>
        <p>Select a policy document from the dropdown and enter your question to begin</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("© 2025 UDST Policy Assistant | For official policy documents, always consult the original sources")