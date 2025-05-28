import os
import nest_asyncio
import json

from autogen import (
    ConversableAgent,
    AssistantAgent,
    UserProxyAgent,
    GroupChat,
    GroupChatManager,
    config_list_from_json
)

from autogen.agentchat.contrib.graph_rag.document import Document, DocumentType
from autogen.agentchat.contrib.graph_rag.neo4j_graph_query_engine import Neo4jGraphQueryEngine
from autogen.agentchat.contrib.graph_rag.neo4j_graph_rag_capability import Neo4jGraphCapability
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent

from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json

# ✅ Config Setup
nest_asyncio.apply()
config_list = config_list_from_json("OAI_CONFIG_LIST", filter_dict={"model": ["gpt-4o"]})

# ✅ Optional Web Search Setup (if Serper enabled)
try:
    from crewai_tools import SerperDevTool
    search_tool = SerperDevTool()
except:
    search_tool = None

# ✅ PDF Parsing Function
def parse_pdf(filename: str):
    file_elements = partition_pdf(
        filename=filename,
        strategy="hi_res",
        languages=["eng"],
        infer_table_structure=True,
        extract_images_in_pdf=True,
        extract_image_block_output_dir="./parsed_pdf_info",
        extract_image_block_types=["Image", "Table"],
        extract_forms=False,
        form_extraction_skip_tables=False,
    )
    elements_to_json(elements=file_elements, filename="parsed_elements.json", encoding="utf-8")
    return file_elements


# ✅ Connect to Neo4j KG
query_engine = Neo4jGraphQueryEngine(
    username="neo4j",
    password="2ViKyTzQ_kQO9njXaUvYyHaavDec8iqBQ9h8iv5NzYs",
    host="neo4j+s://02645b0d.databases.neo4j.io",
    # port=7687,
    database="neo4j",
    llm=config_list,
    embedding={"model_name": "text-embedding-3-small"}
)

query_engine.connect_db()

# ✅ Load sample data (skip parsing again if already parsed)
documents = [Document(doctype=DocumentType.JSON, path_or_url="parsed_elements.json")]
query_engine.init_db(input_doc=documents)

# ✅ Define Agents
llm_config = {
    "config_list": config_list,
    "timeout": 120,
    "temperature": 0.7,
}

user_proxy = UserProxyAgent(
    name="user_proxy",
    system_message="You are the user interface. Pass all requests to the assistant.",
    human_input_mode="ALWAYS",
)

general_assistant = ConversableAgent(
    name="general_assistant",
    llm_config=llm_config,
    system_message="You answer general questions. If document content is needed, ask the RAG agent. If you're unsure, ask the WebSearchAgent.",
    tools=[search_tool] if search_tool else [],
)

web_search_agent = AssistantAgent(
    name="web_search_agent",
    system_message="You search the web when the assistant can't answer a question.",
    tools=[search_tool] if search_tool else [],
    llm_config=llm_config,
    human_input_mode="NEVER"
)

# 🔍 Add Graph RAG
rag_agent = ConversableAgent(
    name="rag_agent",
    human_input_mode="NEVER",
)
Neo4jGraphCapability(query_engine).add_to_agent(rag_agent)

# 🖼️ Image-to-Table Agent (for image-to-markdown tables)
image2table = MultimodalConversableAgent(
    name="image2table_convertor",
    system_message="""
    You convert table images to markdown. Extract and format the table fully. Fix typos if needed. Output clean markdown tables only.
    """,
    llm_config={"config_list": config_list, "max_tokens": 300},
    max_consecutive_auto_reply=1,
    human_input_mode="NEVER"
)

# 🧠 Final Synthesizer
final_synthesizer = AssistantAgent(
    name="final_answer_agent",
    system_message="Based on all context, answer the user's original question in detail.",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

# 🤖 GroupChat Coordination
groupchat = GroupChat(
    agents=[
        user_proxy,
        general_assistant,
        web_search_agent,
        rag_agent,
        image2table,
        final_synthesizer,
    ],
    messages=[],
    speaker_selection_method="round_robin",
)

manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# ✅ Chat Trigger
user_proxy.initiate_chat(
    manager,
    message="I just uploaded a document. What are the key financial highlights of Nvidia's 2024 report?",
)
