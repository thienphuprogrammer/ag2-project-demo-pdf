import os
import sys

# Patch to prevent PyTorch inspection issues
os.environ['STREAMLIT_SERVER_RUN_ON_SAVE'] = 'false'
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

# Add this before any other imports
if 'torch' in sys.modules:
    import torch
    torch.utils.data.DataLoader = None  # Prevent DataLoader from being inspected

import streamlit as st
import tempfile
import logging
import nest_asyncio
from typing import List, Dict, Any

from src.utils import parse_pdf, ensure_directory_exists
from src.config import load_config
from src.neo4j_client import init_query_engine
from src.agents.rag import create_rag_agent
from src.agents.general import create_general_agent
from src.agents.search import create_search_agent
from src.agents.multimodal import create_multimodal_agent
from autogen import UserProxyAgent, GroupChat, GroupChatManager
from autogen.agentchat.contrib.graph_rag.neo4j_graph_rag_capability import Neo4jGraphCapability

nest_asyncio.apply()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize app state
if "initialized" not in st.session_state:
    st.session_state.config_list = load_config()
    st.session_state.messages = []
    st.session_state.temp_dir = tempfile.mkdtemp()
    st.session_state.output_json = os.path.join(st.session_state.temp_dir, "parsed.json")
    st.session_state.image_dir = os.path.join(st.session_state.temp_dir, "images")
    st.session_state.query_engine = None
    st.session_state.agents = None
    st.session_state.manager = None
    st.session_state.initialized = True

def setup_agents() -> UserProxyAgent:
    """Initialize and configure all agents for the chat system.
    
    Returns:
        UserProxyAgent: The configured user proxy agent.
    """
    try:
        # Create user proxy with basic configuration
        user_proxy = UserProxyAgent(
            name="user_proxy",
            system_message="A human admin. Interact with the assistant agents to get information.",
            code_execution_config=False,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=4
        )

        # Add RAG capability if query engine is available
        if st.session_state.query_engine:
            try:
                graph_rag_capability = Neo4jGraphCapability(st.session_state.query_engine)
                graph_rag_capability.add_to_agent(user_proxy)
                logger.info("Added RAG capability to user proxy")
            except Exception as e:
                logger.error(f"Failed to add RAG capability: {e}")
        
        # Create specialized agents with appropriate configurations
        general = create_general_agent(
            st.session_state.config_list,
            name="general_agent",
            human_input_mode="NEVER"
        )
        
        search = create_search_agent(
            st.session_state.config_list,
            name="search_agent",
            human_input_mode="NEVER"
        )
        
        multimodal = create_multimodal_agent(
            st.session_state.config_list,
            name="multimodal_agent",
            human_input_mode="NEVER"
        )
        
        # Create a final agent for summarization
        summarizer = create_general_agent(
            st.session_state.config_list,
            name="summarizer",
            human_input_mode="NEVER",
            system_message="""You are a summarizer. Your job is to provide concise summaries of the conversation 
            and ensure the final response is clear and helpful. Focus on the key points and next steps.
            Keep your responses brief and to the point."""
        )

        # Configure the group chat
        agents = [user_proxy, general, search, summarizer]
        
        # Define the group chat with a clear termination message
        group_chat = GroupChat(
            agents=agents,
            messages=[],
            max_round=4,  # Prevent infinite loops
            speaker_selection_method="round_robin"
        )
        
        # Create the manager with LLM configuration
        manager = GroupChatManager(
            groupchat=group_chat,
            llm_config={
                "config_list": st.session_state.config_list,
                "timeout": 60,
                "cache_seed": 42,
                "temperature": 0.7
            }
        )

        # Store in session state
        st.session_state.agents = agents
        st.session_state.manager = manager
        
        logger.info("Agents initialized successfully")
        return user_proxy
        
    except Exception as e:
        logger.error(f"Error setting up agents: {e}")
        st.error(f"Failed to initialize agents: {e}")
        raise



def refresh_rag_after_upload():
    try:
        ensure_directory_exists(st.session_state.image_dir)
        parse_pdf(
            st.session_state.latest_pdf,
            st.session_state.output_json,
            image_output_dir=st.session_state.image_dir
        )
        st.session_state.query_engine = init_query_engine(st.session_state.config_list, st.session_state.output_json)
        setup_agents()
        st.success("PDF parsed and knowledge base updated.")
    except Exception as e:
        st.error(f"Failed to update knowledge base: {e}")


# UI Layout
st.title("💬 Multi-Agent Chat")

# Sidebar for document upload
with st.sidebar:
    st.header("📄 Document Management")
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
    
    if uploaded_pdf and st.button("Add to Knowledge Base"):
        with st.spinner("Processing document..."):
            try:
                # Save the uploaded file
                pdf_path = os.path.join(st.session_state.temp_dir, uploaded_pdf.name)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_pdf.read())
                
                # Update session state and refresh RAG
                st.session_state.latest_pdf = pdf_path
                refresh_rag_after_upload()
                st.success("Document processed successfully!")
                
                # Re-initialize agents with the new document
                st.session_state.user_proxy = setup_agents()
                
            except Exception as e:
                st.error(f"Error processing document: {e}")
                logger.exception("Error in document processing")
    
    # Display system status
    st.markdown("---")
    st.subheader("System Status")
    st.write(f"Agents initialized: {bool(st.session_state.get('agents'))}")
    st.write(f"Document loaded: {bool(st.session_state.get('query_engine'))}")

# Initialize chat if not already done
if 'user_proxy' not in st.session_state:
    st.session_state.user_proxy = setup_agents()

# Main chat interface
st.subheader("🧠 Chat with Analysts")

# Display chat history
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your analysis assistant. How can I help you today?"}
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response from agents
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            # Initialize chat if needed
            if not st.session_state.get('chat_initiated', False):
                st.session_state.chat_initiated = True
                        
            # Get the response from the agent system
            try:
                # Initiate the chat and collect the full response
                teacher = st.session_state.agents[0]
                response = teacher.run(
                    recipient=st.session_state.manager,
                    message=prompt,
                    summary_method="reflection_with_llm"
                )
                
                response.process()
                full_response = response.messages[-1]["content"]

                
            except Exception as e:
                logger.error(f"Error in agent response: {e}")
                full_response = f"I encountered an error: {str(e)}"
            
            # Update the final response without the cursor
            response_placeholder.markdown(full_response)
            
            # Add the complete response to the chat history
            if full_response.strip():
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            st.error(error_msg)
            logger.exception("Error in chat processing")
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
