# orchestrator.py
from typing import TypedDict, Annotated, List
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver  # Checkpointer pour la mémoire
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

# Importation des outils et du RAG
from agent_tools import EXTERNAL_TOOLS
from finance_rag import run_rag_tool


# --- 1. Définition de l'État du Graphe ---

class AgentState(TypedDict):
    """État minimal requis pour la mémoire et le routage LangGraph."""
    messages: Annotated[List[BaseMessage], add]
    thread_id: str


# --- 2. Initialisation du LLM et des Outils ---

# 1. Liste de tous les outils (RAG inclus)
ALL_TOOLS: List[BaseTool] = EXTERNAL_TOOLS + [run_rag_tool]

# 2. Initialisation du LLM avec tous les outils bindés (Tool Calling)
llm_with_all_tools = ChatOpenAI(model="gpt-4o-mini", temperature=0.0).bind_tools(ALL_TOOLS)

# 🛑 CORRECTION : Création du map d'outils robuste pour l'exécution 🛑
TOOL_MAP = {}
for tool in ALL_TOOLS:
    # Pour les outils définis avec @tool (comme le RAG), on utilise .func
    if hasattr(tool, 'func'):
        TOOL_MAP[tool.name] = tool.func
    # Pour les objets de classe (comme TavilySearch), on cherche .run ou .invoke
    elif hasattr(tool, 'run'):
        TOOL_MAP[tool.name] = tool.run
    elif hasattr(tool, 'invoke'):
        TOOL_MAP[tool.name] = tool.invoke
    else:
        # Mesure de sécurité si un outil inconnu est présent
        raise AttributeError(
            f"Impossible de trouver la méthode d'exécution (.func, .run, ou .invoke) pour l'outil {tool.name}. Vérifiez la définition.")


# --- 3. Définition des Nœuds (Actions) du Graphe ---

def run_llm_router(state: AgentState):
    """Nœud 1 : Le LLM décide de la prochaine action (réponse directe ou appel d'outil)."""
    # LangGraph passe l'historique complet via state["messages"]
    result = llm_with_all_tools.invoke(state["messages"])
    return {"messages": [result]}


def execute_tool(state: AgentState):
    """Nœud 2 : Exécute l'outil sélectionné par le LLM et retourne les ToolMessage."""
    last_message = state["messages"][-1]  # AIMessage contenant les tool_calls
    tool_results = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        args = tool_call["args"]

        # Utilisation du map prédéfini pour trouver la fonction
        tool_function = TOOL_MAP.get(tool_name)

        if tool_function:
            try:
                # Exécution de la fonction/outil
                output = tool_function(**args)

                # Ajout du résultat en tant que ToolMessage
                tool_results.append(
                    ToolMessage(
                        content=str(output),  # S'assurer que le contenu est une chaîne
                        tool_call_id=tool_call["id"],
                        name=tool_name
                    )
                )
            except Exception as e:
                # Gestion des erreurs pour éviter le blocage du graphe
                tool_results.append(
                    ToolMessage(
                        content=f"Erreur d'exécution de l'outil {tool_name}: {e}",
                        tool_call_id=tool_call["id"],
                        name=tool_name
                    )
                )
        else:
            tool_results.append(
                ToolMessage(
                    content=f"Outil {tool_name} inconnu. Vérifiez agent_tools.py et finance_rag.py.",
                    tool_call_id=tool_call["id"],
                    name=tool_name
                )
            )

    return {"messages": tool_results}


# --- 4. Définition de la Logique de Routage ---

def should_continue(state: AgentState) -> str:
    """Détermine la prochaine étape : 'tool' (boucle) ou 'end' (réponse finale)."""
    last_message = state["messages"][-1]

    # Si le LLM a renvoyé des tool_calls, nous devons exécuter l'outil.
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool"

    # Sinon, le LLM a donné la réponse finale, et nous terminons.
    return "end"


# --- 5. Construction du Graphe (LangGraph) ---

workflow = StateGraph(AgentState)
workflow.add_node("llm", run_llm_router)
workflow.add_node("tool", execute_tool)

# Définition de l'entrée et des transitions
workflow.set_entry_point("llm")
workflow.add_conditional_edges("llm", should_continue, {"tool": "tool", "end": END})
workflow.add_edge("tool", "llm")  # Après l'exécution de l'outil, on revient toujours au LLM

# Compilation du graphe avec Checkpointer pour la gestion de la mémoire
MEMORY = InMemorySaver()
app = workflow.compile(checkpointer=MEMORY)


# --- 6. Fonction d'Orchestration pour l'Interface ---

def run_orchestrator(question: str, thread_id: str) -> str:
    """Lance l'exécution du LangGraph avec la question et la mémoire (thread_id)."""

    # Configuration de la session (mémoire) LangGraph
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    new_message = HumanMessage(content=question)

    # Invocation du graphe
    final_state = app.invoke({"messages": [new_message]}, config=config)

    # Récupère le contenu de la dernière réponse (qui est toujours le message final du LLM)
    final_response = final_state["messages"][-1].content

    return final_response