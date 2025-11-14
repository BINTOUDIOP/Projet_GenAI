# tools.py
import os
import requests
import ast
import operator as op
import json
from dotenv import load_dotenv
from typing import List, Dict, Any

# Modules LangChain modernes
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

# --- Configuration et Initialisation ---
load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
CALENDAR_FILE = "./data/calendar.json"

# --- Outil 1 : Calculatrice Sécurisée ---

# Opérateurs autorisés pour l'évaluation sécurisée
ALLOWED_OPERATORS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg
}


def safe_eval(expression: str) -> str:
    """Exécute une expression arithmétique de manière sécurisée."""
    expression = expression.replace("^", "**")  # Remplacement pour la syntaxe Python
    try:
        node = ast.parse(expression, mode="eval").body

        def _eval(n):
            if isinstance(n, (ast.Num, ast.Constant)): return n.value
            if isinstance(n, ast.BinOp):
                op_type = type(n.op)
                if op_type in ALLOWED_OPERATORS:
                    return ALLOWED_OPERATORS[op_type](_eval(n.left), _eval(n.right))
            if isinstance(n, ast.UnaryOp) and type(n.op) in ALLOWED_OPERATORS:
                return ALLOWED_OPERATORS[type(n.op)](_eval(n.operand))
            raise ValueError("Opération non supportée ou non sécurisée.")

        return str(_eval(node))
    except Exception as e:
        return f"Erreur calcul: Veuillez vérifier l'expression. Erreur: {e}"


@tool
def calculate_financial_operation(expression: str) -> str:
    """
    Exécute une expression mathématique simple pour des calculs financiers (addition, soustraction, multiplication, division, puissance).
    Fournir l'expression complète sous forme de chaîne (ex: '1500 * (1 + 0.05)^3').
    """
    return safe_eval(expression)


# --- Outil 2 : Recherche Web (Tavily) ---

# TavilySearchResults est déjà une classe Tool/Runnable
web_search_tool = TavilySearch(
    name="recherche_web_actualites",
    description="Utilisez cet outil pour trouver des informations externes, des actualités financières récentes, ou des définitions qui ne sont pas dans les documents internes.",
    max_results=3
)


# --- Outil 3 : Météo (OpenWeatherMap) ---

@tool
def get_weather_for_city(city: str) -> str:
    """
    Récupère la météo actuelle pour une ville donnée.
    Utilisez cet outil seulement si l'utilisateur pose une question sur la météo.
    """
    if not OPENWEATHER_API_KEY:
        return "Erreur : La clé API OpenWeatherMap est manquante. Météo simulée: Il fait 20°C et ensoleillé à Paris."

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric&lang=fr"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        main_desc = data['weather'][0]['description']
        temp = data['main']['temp']

        return f"Météo à {city.capitalize()} : {main_desc}, température actuelle de {temp}°C."
    except requests.exceptions.HTTPError as errh:
        return f"Erreur HTTP: La ville '{city}' n'a pas été trouvée ou service indisponible."
    except requests.exceptions.RequestException:
        return "Erreur de connexion à l'API météo. Veuillez réessayer."


# --- Outil 4 : Calendrier / Todo List Locale ---

@tool
def read_calendar(query: str) -> str:
    """
    Recherche dans un calendrier local (data/calendar.json) pour trouver des événements, des rendez-vous ou des tâches.
    Fournir la date (ex: 'aujourd'hui', '25 mars') ou le sujet (ex: 'réunion budget').
    """
    if not os.path.exists(CALENDAR_FILE):
        return f"Aucun calendrier local trouvé ({CALENDAR_FILE} absent). Veuillez créer le fichier."
    try:
        with open(CALENDAR_FILE, "r", encoding="utf-8") as f:
            events = json.load(f)

        q = query.lower().strip()

        # Logique de recherche simple par mots-clés
        matches = [
            e for e in events
            if q in e.get("title", "").lower() or q in e.get("date", "").lower()
        ]

        if not matches:
            return "Aucun événement correspondant dans le calendrier local."

        return json.dumps(matches, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Erreur lors de la lecture du calendrier: {e}"


# --- Liste des Outils Exportée ---

# NOTE : La recherche web est ajoutée directement car elle est déjà une classe Tool
EXTERNAL_TOOLS = [
    calculate_financial_operation,
    web_search_tool,
    get_weather_for_city,
    read_calendar
]