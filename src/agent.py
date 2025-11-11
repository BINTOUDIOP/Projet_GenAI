import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from tools import tool_calcul, tool_meteo, tool_web_search

load_dotenv()
llm = ChatOpenAI(model=os.getenv("CHAT_MODEL","gpt-4o-mini"), temperature=0)

TOOLS_DESC = """Tu peux utiliser des OUTILS si besoin.
Outils disponibles (tu renvoies UNIQUEMENT le résultat brut de l'outil choisi):
- CALCUL(expression)
- METEO(ville)
- WEB(query)
Si aucun outil n'est nécessaire, réponds directement.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", TOOLS_DESC + "Réponds toujours en français."),
    ("human", "{question}")
])

def run_tool(tool_call: str) -> str:
    # Très simple parsing: TOOL(args)
    try:
        name, args = tool_call.strip().split("(", 1)
        args = args.rsplit(")", 1)[0]
        if name.upper()=="CALCUL":
            return tool_calcul(args)
        if name.upper()=="METEO":
            return tool_meteo(args)
        if name.upper()=="WEB":
            return tool_web_search(args, n=5)
    except:
        pass
    return None

def answer_with_agent(question: str) -> str:
    # Heuristique: le LLM propose soit un texte direct, soit "TOOL(args)" en 1ère ligne
    resp = llm.invoke(PROMPT.format_messages(question=question)).content.strip()
    if "(" in resp and resp.split("(",1)[0].upper() in {"CALCUL","METEO","WEB"}:
        tool_out = run_tool(resp)
        return tool_out if tool_out else f"(Outil non reconnu) {resp}"
    return resp