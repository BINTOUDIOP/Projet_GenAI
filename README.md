# Assistant Intelligent Multi-Compétences (RAG + Agents)

Ce projet implémente un assistant conversationnel basé sur l'IA, capable d'orchestrer plusieurs compétences : la recherche d'informations dans des documents internes (**RAG** - Retrieval-Augmented Generation) et l'utilisation d'outils externes (**Agents** - Calcul, Météo, Web).

L'architecture utilise **LangChain** et l'**Agent Executor** basé sur les fonctions OpenAI pour le raisonnement et la sélection d'outils, le tout exposé via une interface **Chainlit**.

---

##  Architecture du Projet

L'architecture est construite autour de l'outil **LangGraph**, qui gère l'état et le flux de travail de l'Agent. Nous utilisons une structure modulaire avec le principe de la **séparation des préoccupations**.

| Fichier | Rôle | Technologie Clé |
| :--- | :--- | :--- |
| **`finance_rag.py`** | **Capacité RAG** | **ChromaDB**, **LCEL**, `@tool` |
| **`agent_tools.py`** | **Bibliothèque d'Outils Externes** | `@tool`, TavilySearch, Requests |
| **`orchestrator.py`** | **Cœur de l'Agent & Routage** | **LangGraph**, **Tool Calling** (GPT-4o-mini), InMemorySaver |
| **`app_chainlit.py`** | **Interface Utilisateur (UX)** | Chainlit, Isolation de Session (Mémoire unique par utilisateur) |

---

##  Installation et Lancement

### 1. Pré-requis

* **Python :** Version 3.11
* **Clés API :** Une clé **OpenAI API Key** est obligatoire. Une clé **Tavily API Key** est nécessaire pour la recherche web.
* **Clés Google Cloud** pour le déploiement :  
  -`GCP_PROJECT_ID` : ID de ton projet Google Cloud  
  -`GCP_REGION` : Région de déploiement Cloud Run  
  -`GCP_SA_KEY` : Fichier JSON de ta **Service Account Key** 

### 2. Configuration de l'Environnement

Créez un fichier **`.env`** à la racine du projet pour stocker toutes les configurations sensibles.

```bash
# .env
OPENAI_API_KEY="votre_clé_openai_ici"
TAVILY_API_KEY="votre_clé_tavily_ici"
# Optionnel : pour l'outil météo
OPENWEATHER_API_KEY="votre_clé_meteo_ici" 

# Configurations par défaut (modifiables)
LLM_MODEL=gpt-4o-mini
CHROMA_DB_DIR=./chroma_db_agent
DOCUMENT_PATH=./data/finance_document.pdf
```
### 3. Installation des Dépendances

Créez un fichier **`requirements.txt`** listant toutes les bibliothèques nécessaires, puis exécutez l'installation.

```bash
pip install -r requirements.txt
```
### 4. Ingestion des Documents (RAG)
Le processus d'ingestion est géré automatiquement au premier lancement (ou lors de l'importation de finance_rag.py). Assurez-vous que le document PDF de référence (finance_document.pdf) est placé dans le dossier ./data.

### 5. Lancement de l'Application
Lancez l'interface web via Chainlit :
l'application sera accessible via l'URL affichée dans le terminal (généralement http://localhost:8000).
```bash
chainlit run app_chainlit.py -w
```

### 5. Déploiement sur le cloud
Le projet a été déployé temporairement sur Google Cloud Run via GitHub Actions :
- L’URL de l’application est accessible pour tester l’assistant à distance mais elle sera désactivée après la période de test.https://agent-orchestre-finance-750867412054.europe-west9.run.app/
- logs et erreurs sont monitorés via Cloud Run pour assurer la stabilité durant la période active.


# Instructions de Test
L'Agent Orchestrateur est capable de choisir l'outil approprié. Testez les différents modes :

| Scénario | Exemple de Requête                                             | Outil (via Agent) |
| :--- |:---------------------------------------------------------------| :--- |
| **RAG** | *Quelle est la politique de congés selon le manuel ?*          | `Document_QandA` |
| **Calcul** | *Calcule 15% de 2500.*                                         | `calculator` |
| **Web Search** | *Qui a remporté la dernière élection présidentielle aux USA ?* | `web_search` |
| **Météo** | *Donne la météo pour Londres, UK.*                             | `weather` |
| **Mémoire** | *Mon nom est Alex.* (Puis) *Quel est mon nom ?*                | *Aucun outil (LLM + Mémoire)* |



N'hésitez pas à nous contacter pour toute question sur l'architecture ou les choix techniques( Bintou DIOP et Ismael Coulibaly).

**Félicitations pour la complétion de votre Assistant Intelligent !**