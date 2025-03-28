import logging
import os
import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.config import Config
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from langchain_community.chat_models import ChatDatabricks
from langchain_openai import ChatOpenAI
from databricks_langchain.genie import GenieAgent
from datetime import datetime
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
import json
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Databricks Workspace Client
w = WorkspaceClient()

# Ensure environment variable is set correctly
assert os.getenv('SERVING_ENDPOINT'), "SERVING_ENDPOINT must be set in app.yaml."

def get_user_info():
    headers = st.context.headers
    return dict(
        user_name=headers.get("X-Forwarded-Preferred-Username"),
        user_email=headers.get("X-Forwarded-Email"),
        user_id=headers.get("X-Forwarded-User"),
    )

user_info = get_user_info()

# --- MODELO LLM COM FERRAMENTAS ---
#llm = ChatDatabricks(endpoint=os.environ["SERVING_ENDPOINT"], temperature=0.7)
llm = ChatOpenAI(model="gpt-4o", temperature=0.7, verbose=True)
tool = TavilySearchResults(max_results=3)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)

# --- GENIE AGENT (adicionado) ---
GENIE_SPACE_ID = os.getenv("GENIE_SPACE_ID")
genie_client = WorkspaceClient(
    config=Config(
        host=os.getenv("DATABRICKS_HOST"),
        token=os.getenv("DATABRICKS_GENIE_PAT"),
        auth_type="pat"
    )
)
genie_agent = GenieAgent(
    genie_space_id=GENIE_SPACE_ID,
    genie_agent_name="Genie",
    description="Agente Genie para responder com acesso ao espaço Genie.",
    client=genie_client
)

# --- TIPO DO ESTADO DO LANGGRAPH ---
class State(TypedDict):
    messages: Annotated[list, add_messages]

# --- NODE PRINCIPAL ---
def chatbot_node(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"]) ]}

# --- NODE GENIE ---
def genie_node(state: State):
    response = genie_agent.invoke(state)
    return {"messages": [response["messages"][-1]]}

# --- NODE CODER ---
def coder_node(state: State):
    prompt = (
        "Você é um assistente que interpreta respostas de agentes que acessam dados estruturados. "
        "Abaixo está o histórico da conversa e a resposta mais recente do agente Genie:\n\n"
        "Responda ao usuário de forma clara, explicando o que os dados significam em relação à pergunta original.\n\n"
    )

    # Filtrar mensagens que não contenham tool_calls e que não sejam ToolMessages
    filtered_history = []
    skip_next_tool_call = False

    for msg in state["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            # Ignora esta e a próxima ToolMessage
            skip_next_tool_call = True
            continue
        if isinstance(msg, ToolMessage):
            if skip_next_tool_call:
                skip_next_tool_call = False
                continue
        filtered_history.append(msg)

    response = llm.invoke([{"role": "user", "content": prompt}] + filtered_history)
    return {"messages": [response]}

# --- NODE DE FERRAMENTAS ---
class BasicToolNode:
    def __init__(self, tools):
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        message = inputs["messages"][-1]
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"]
            ))
        return {"messages": outputs}

# --- ROTEAMENTO ---
def route_decision(state: State):
    last_message = state["messages"][-1]
    content = getattr(last_message, "content", "").lower()
    if "genie" in content or "consulta" in content or "dados do espaço" in content:
        return "genie"
    elif hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# --- BUILDER DO LANGGRAPH ---
builder = StateGraph(State)
builder.add_node("chatbot", chatbot_node)
builder.add_node("tools", BasicToolNode(tools))
builder.add_node("genie", genie_node)
builder.add_node("coder", coder_node)
builder.add_conditional_edges("chatbot", route_decision, {"tools": "tools", "genie": "genie", END: END})
builder.add_edge("tools", "chatbot")
builder.add_edge("genie", "coder")
builder.add_edge("coder", END)
builder.set_entry_point("chatbot")
graph = builder.compile(checkpointer=MemorySaver())

# --- INICIO STREAMLIT ---
st.set_page_config(page_title="LangGraph Chat")

if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

def start_new_chat():
    timestamp = datetime.now().strftime("%d/%m %H:%M:%S")
    chat_id = f"chat_{timestamp}"
    st.session_state.current_chat = chat_id
    st.session_state.conversations[chat_id] = {
        "name": f"Conversa {timestamp}",
        "messages": [],
        "raw_messages": [],
        "thread_id": chat_id
    }

# Sidebar
st.sidebar.title("💬 Conversas")
if st.sidebar.button("➕ Nova Conversa"):
    start_new_chat()
for cid, data in st.session_state.conversations.items():
    if st.sidebar.button(data["name"]):
        st.session_state.current_chat = cid

st.title("🤖 Chat com LangGraph + Tavily + Genie")

if st.session_state.current_chat:
    chat_id = st.session_state.current_chat
    chat_data = st.session_state.conversations[chat_id]
    st.subheader(chat_data["name"])

    for msg in chat_data["messages"]:
        st.markdown(f"**Você:** {msg['user']}")
        st.markdown(f"**Bot:** {msg['bot']}")

    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Digite sua mensagem:", key="chat_input")
        submitted = st.form_submit_button("Enviar")

    if submitted and user_input:
        chat_data["raw_messages"].append({"type": "human", "content": user_input})

        state = {"messages": chat_data["raw_messages"]}
        new_ai_messages = []

        for event in graph.stream(
            state,
            config={"configurable": {
                "thread_id": chat_id,
                "checkpoint_id": chat_id,
                "checkpoint_ns": "default_ns"}}
        ):
            for value in event.values():
                ai_msg = value["messages"][-1]
                new_ai_messages.append(ai_msg)

        chat_data["raw_messages"].extend(new_ai_messages)
        chat_data["messages"].append({
            "user": user_input,
            "bot": new_ai_messages[-1].content if new_ai_messages else "(erro na resposta)"
        })

        st.rerun()
else:
    st.info("Selecione ou inicie uma nova conversa.")
