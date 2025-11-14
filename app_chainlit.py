import chainlit as cl
import os
from dotenv import load_dotenv
from chainlit import user_session
from orchestrator import run_orchestrator

load_dotenv()


@cl.on_chat_start
async def start():
    # 1. Configuration de la mémoire (thread_id unique pour cette session)
    thread_id = cl.user_session.get("id")
    cl.user_session.set("thread_id", thread_id)

    # 2. Vérification du statut de la mémoire interne (RAG) pour affichage professionnel
    try:
        # Importation nécessaire pour vérifier si la chaîne RAG a pu être construite
        from finance_rag import RAG_CHAIN

        if RAG_CHAIN:
            rag_status = "🟢 La **mémoire**  est active."
        else:
            rag_status = "🟠 Erreur de chargement de la mémoire interne. (Fichier PDF manquant ?)"

    except Exception:
        # Ceci capture les erreurs d'initialisation critique (ex: clé API OpenAI)
        rag_status = "🔴 Erreur critique lors de l'initialisation des sources internes."

    # 3. Message de bienvenue avec statut et exemples
    await cl.Message(
        content=f" **Assistant Financier Multi Compétences**\n\n"
                f"Bonjour ! Je peux répondre à vos questions, faire des calculs précis et chercher des informations en temps réel.\n\n"
                f"{rag_status}",
        author="Assistant",
    ).send()

    await cl.Message(
        content="Voici quelques exemples de questions que vous pouvez me poser :\n"
                "Quelle est la définition de la monnaie selon nos documents ?\n"
                "Calculez 5000 * (1 + 0.045)^5\n"
                "Qui est le CEO de LVMH ?\n"
                "Quels événements sont prévus pour ce week-end ?",
        author="Assistant",
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """Fonction appelée à chaque nouveau message utilisateur."""

    # 1. Récupérer le thread_id unique stocké dans la session
    current_thread_id = user_session.get("thread_id")

    # 2. Exécution de l'Orchestrateur LangGraph
    response = await cl.make_async(run_orchestrator)(message.content, current_thread_id)

    # 3. Affichage de la réponse finale de l'Agent
    await cl.Message(
        content=response,
        author="Assistant"
    ).send()