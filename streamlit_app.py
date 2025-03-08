import streamlit as st
import os
import numpy as np
import faiss
import json
from mistralai import Mistral, UserMessage
import time

# Set page configuration
st.set_page_config(
    page_title="UDST Policy Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    .header-container {
        padding: 1rem;
        background-color: #003366;
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    /* Card styling */
    .card {
        border-radius: 10px;
        padding: 1.5rem;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    /* Answer container */
    .answer-container {
        border-radius: 10px;
        padding: 1.5rem;
        background-color: #e8f4f8;
        border-left: 5px solid #3498db;
        margin-top: 1rem;
    }
    
    /* Sources container */
    .source-container {
        background-color: #f9f9f9;
        border-radius: 8px;
        padding: 0.5rem;
        margin-top: 0.5rem;
        border: 1px solid #eee;
    }
    
    /* Policy icons */
    .policy-icon {
        font-size: 1.5rem;
        margin-right: 0.5rem;
        color: #003366;
    }
    
    /* Section headers */
    .section-header {
        color: #003366;
        font-weight: bold;
        border-bottom: 2px solid #eee;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #003366;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #004080;
    }
    
    /* Query box styling */
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 2px solid #eee;
        padding: 0.5rem;
    }
    
    /* Logo styling */
    .logo-title {
        display: flex;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# API key (hardcoded)
api_key = "LoXIODO6VkldB64uwva76l1zDpIz6cfu"

# Helper functions
def get_safe_filename(policy_name):
    return policy_name.lower().replace(" ", "_")

def get_text_embedding(list_txt_chunks, api_key):
    client = Mistral(api_key=api_key)
    embeddings_batch_response = client.embeddings.create(
        model="mistral-embed",
        inputs=list_txt_chunks
    )
    return embeddings_batch_response.data

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
        # Fallback to small model
        try:
            chat_response = client.chat.complete(
                model="mistral-small-latest",
                messages=messages,
            )
            return chat_response.choices[0].message.content
        except:
            return f"Error generating response: {str(e)}"

# Policy URLs and icons
POLICY_INFO = {
    "Sport and Wellness Facilities": {
        "url": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/sport-and-wellness-facilities-and",
        "icon": "🏆"
    },
    "Credit Hour Policy": {
        "url": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/credit-hour-policy",
        "icon": "⏱️"
    },
    "Final Grade Policy": {
        "url": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/final-grade-policy",
        "icon": "🎓"
    },
    "Student Appeals Policy": {
        "url": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-appeals-policy",
        "icon": "⚖️"
    },
    "Student Attendance Policy": {
        "url": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-attendance-policy",
        "icon": "📋"
    },
    "Student Counselling Services": {
        "url": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/student-counselling-services-policy",
        "icon": "💬"
    },
    "Library Space Policy": {
        "url": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/use-library-space-policy",
        "icon": "📚"
    },
    "Transfer Policy": {
        "url": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/transfer-policy",
        "icon": "🔄"
    },
    "Academic Schedule Policy": {
        "url": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/academic-schedule-policy",
        "icon": "📅"
    },
    "Student Conduct Policy": {
        "url": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-conduct-policy",
        "icon": "📝"
    }
}

# Get available policies
def get_available_policies():
    if not os.path.exists("indexes"):
        return []
    
    available = []
    for policy in POLICY_INFO.keys():
        safe_name = get_safe_filename(policy)
        if os.path.exists(f"indexes/{safe_name}.index") and os.path.exists(f"chunks/{safe_name}_chunks.json"):
            available.append(policy)
    
    return available

# Sidebar
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>UDST Policy Assistant</h2>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; margin-bottom: 2rem;'><img src='https://www.udst.edu.qa/wp-content/uploads/2018/04/UDST-Doha.png' width='180'/></div>", unsafe_allow_html=True)
    
    st.markdown("<p class='section-header'>Available Policies</p>", unsafe_allow_html=True)
    
    available_policies = get_available_policies()
    
    if not available_policies:
        st.warning("⚠️ No policy data available. Please run preprocessing first.")
    else:
        st.success(f"✅ {len(available_policies)} policies available")
    
    st.markdown("<p class='section-header'>About</p>", unsafe_allow_html=True)
    st.markdown("""
    This assistant helps you find information about UDST policies using 
    Retrieval Augmented Generation (RAG) technology.
    
    Simply select a policy, ask a question, and get accurate answers with source references.
    """)
    
    st.markdown("<p class='section-header'>Created By</p>", unsafe_allow_html=True)
    st.markdown("Your Name - UDST Assignment")

# Main Content
st.markdown("<div class='header-container'><h1>UDST Policy Questions & Answers</h1><div>University of Doha for Science and Technology</div></div>", unsafe_allow_html=True)

# Check if data is available
if len(get_available_policies()) == 0:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.warning("⚠️ No policy data found. Please run the preprocessing script first.")
    st.markdown("</div>", unsafe_allow_html=True)
else:
    # Two-column layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='section-header'>Select a Policy</p>", unsafe_allow_html=True)
        
        # Policy selector with icons
        policy_options = {}
        for policy in get_available_policies():
            icon = POLICY_INFO[policy]["icon"]
            policy_options[f"{icon} {policy}"] = policy
        
        selected_policy_display = st.selectbox(
            "Choose a policy to query:",
            options=list(policy_options.keys()),
            label_visibility="collapsed"
        )
        
        selected_policy = policy_options[selected_policy_display] if selected_policy_display else None
        
        if selected_policy:
            st.markdown(f"<p><strong>Selected:</strong> {POLICY_INFO[selected_policy]['icon']} {selected_policy}</p>", unsafe_allow_html=True)
            policy_url = POLICY_INFO[selected_policy]['url']
            st.markdown(f"<p><a href='{policy_url}' target='_blank'>View original policy →</a></p>", unsafe_allow_html=True)
        
        st.markdown("<p class='section-header'>Ask Your Question</p>", unsafe_allow_html=True)
        query = st.text_input("Enter your question about this policy:", placeholder="e.g. What are the requirements for...", label_visibility="collapsed")
        
        if st.button("🔍 Get Answer"):
            if not query:
                st.error("Please enter a question first")
            elif not selected_policy:
                st.error("Please select a policy first")
            else:
                st.session_state['run_query'] = True
                st.session_state['policy'] = selected_policy
                st.session_state['query'] = query
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Recent Questions Card
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='section-header'>Suggested Questions</p>", unsafe_allow_html=True)
        
        # Sample questions for each policy
        sample_questions = {
            "Sport and Wellness Facilities": [
                "What are the opening hours?",
                "How do I register for sports facilities?",
                "What are the fees for using facilities?"
            ],
            "Credit Hour Policy": [
                "How many credits do I need to graduate?",
                "What is the maximum course load per semester?",
                "How are credit hours calculated?"
            ],
            "Final Grade Policy": [
                "How are final grades determined?",
                "What is the process for grade appeals?",
                "When are final grades released?"
            ]
        }
        
        # Display suggestions for selected policy
        if selected_policy:
            questions = sample_questions.get(selected_policy, 
                ["What does this policy cover?", 
                 "What are the main requirements?", 
                 "Who does this policy apply to?"])
            
            for q in questions:
                if st.button(q, key=f"btn_{q}"):
                    st.session_state['run_query'] = True
                    st.session_state['policy'] = selected_policy
                    st.session_state['query'] = q
        else:
            st.info("Select a policy to see suggested questions")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='section-header'>Answer</p>", unsafe_allow_html=True)
        
        # Process query if triggered
        if st.session_state.get('run_query', False):
            policy = st.session_state['policy']
            query = st.session_state['query']
            
            # Reset state
            st.session_state['run_query'] = False
            
            with st.spinner(f"Searching {policy} and generating answer..."):
                try:
                    # Get safe filename
                    safe_name = get_safe_filename(policy)
                    
                    # Check if required files exist
                    chunks_path = f"chunks/{safe_name}_chunks.json"
                    index_path = f"indexes/{safe_name}.index"
                    
                    if not os.path.exists(chunks_path) or not os.path.exists(index_path):
                        st.error(f"Data for {policy} not found. Please run the preprocessing script.")
                    else:
                        # Display query
                        st.info(f"Question: {query}")
                        
                        # Load chunks
                        with open(chunks_path, "r", encoding="utf-8") as f:
                            chunks = json.load(f)
                        
                        # Load index
                        index = faiss.read_index(index_path)
                        
                        # Generate query embedding
                        query_embedding_data = get_text_embedding([query], api_key)
                        query_embedding = np.array([query_embedding_data[0].embedding])
                        
                        # Search for relevant chunks
                        k = min(3, len(chunks))  # Get top 3 or fewer if not enough chunks
                        D, I = index.search(query_embedding, k)
                        
                        # Get retrieved chunks
                        retrieved_chunks = [chunks[i] for i in I.tolist()[0]]
                        context = "\n".join(retrieved_chunks)
                        
                        # Create prompt
                        prompt = f"""
                        Context information about {policy} is below:
                        ---------------------
                        {context}
                        ---------------------
                        
                        Based ONLY on the context information provided and not prior knowledge, 
                        answer the following query about {policy}:
                        
                        Query: {query}
                        
                        Answer:
                        """
                        
                        # Get response from Mistral
                        start_time = time.time()
                        response = query_mistral(prompt, api_key)
                        end_time = time.time()
                        
                        # Display response time
                        response_time = end_time - start_time
                        
                        # Display response
                        st.markdown("<div class='answer-container'>", unsafe_allow_html=True)
                        st.write(response)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        st.caption(f"Response generated in {response_time:.2f} seconds")
                        
                        # Sources
                        st.markdown("<p class='section-header' style='margin-top: 1.5rem;'>Sources</p>", unsafe_allow_html=True)
                        
                        for i, chunk in enumerate(retrieved_chunks):
                            with st.expander(f"Source {i+1} - Relevance: {D.tolist()[0][i]:.2f}"):
                                st.markdown("<div class='source-container'>", unsafe_allow_html=True)
                                st.text(chunk[:400] + "..." if len(chunk) > 400 else chunk)
                                st.markdown("</div>", unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.info("Select a policy and ask a question to get started")
        
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 2rem; padding: 1rem; background-color: #f8f9fa; border-radius: 10px;">
    <p>UDST Policy Assistant | Developed for Assignment Submission<br/>
    Created using RAG (Retrieval Augmented Generation) with Mistral AI
    </p>
</div>
""", unsafe_allow_html=True)