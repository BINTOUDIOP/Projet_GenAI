from langchain_openai import ChatOpenAI
import os, re
from dotenv import load_dotenv
from rag_chain import answer as rag_answer
from agent import answer_with_agent

load_dotenv()
llm = ChatOpenAI(model=os.getenv("CHAT_MODEL","gpt-4o-mini"), temperature=0)

# Classifieur très léger basé sur des règles + LLM de secours
DOC_HINTS = ["selon", "dans le document", "manuel", "politique", "procédure", "rapport", "PDF", "docx"]

def simple_route(query: str) -> str:
    q = query.lower()

    # 1) Heuristiques "Agent"
    if re.search(r"\b(\d+\s*[\+\-\*/]\s*\d+|\bmeteo|météo|temperature|température|recherche sur (le )?web|google|internet)\b", q):
        return "AGENT"

    # 2) Heuristiques "RAG"
    if any(h in q for h in DOC_HINTS):
        return "RAG"

    # 3) LLM fallback très court
    intent = llm.invoke(
        "Tu es un routeur. Réponds par 'RAG', 'AGENT' ou 'SMALLTALK' uniquement.\n"
        f"Question: {query}"
    ).content.strip().upper()
    if intent not in {"RAG","AGENT","SMALLTALK"}:
        intent = "SMALLTALK"
    return intent

def answer(query: str) -> str:
    mode = simple_route(query)
    if mode == "RAG":
        return rag_answer(query)
    if mode == "AGENT":
        return answer_with_agent(query)
    # SMALLTALK
    return "Bonjour ! Comment puis-je t’aider ? 😊" if len(query.split())<=2 else \
           "D’accord ! Dis-m’en un peu plus et je m’adapte."