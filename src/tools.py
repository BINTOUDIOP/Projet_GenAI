import math, requests
from duckduckgo_search import DDGS

# --------- Calculatrice ----------
def tool_calcul(expression: str) -> str:
    try:
        # Sécurisé: autorise chiffres et opérateurs simples
        safe = expression.replace(" ", "")
        if not all(c in "0123456789.+-*/()% " for c in expression):
            return "Opération non autorisée."
        return str(eval(safe, {"__builtins__": {}}, {"math": math}))
    except Exception as e:
        return f"Erreur calcul: {e}"

# --------- Météo (Open-Meteo, sans clé) ----------
def tool_meteo(city: str) -> str:
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "fr"}
        ).json()
        if not geo.get("results"):
            return "Ville introuvable."
        lat = geo["results"][0]["latitude"]; lon = geo["results"][0]["longitude"]
        met = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": True, "timezone": "auto", "daily": "temperature_2m_max,temperature_2m_min"}
        ).json()
        cur = met.get("current_weather", {})
        t = cur.get("temperature")
        w = cur.get("windspeed")
        return f"Météo à {city}: {t}°C, vent {w} km/h (source: Open-Meteo)."
    except Exception as e:
        return f"Erreur météo: {e}"

# --------- Recherche Web (DuckDuckGo) ----------
def tool_web_search(query: str, n: int = 5) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=n))
        if not results:
            return "Aucun résultat."
        lines = []
        for r in results:
            title = r.get("title"); href = r.get("href"); body = r.get("body")
            lines.append(f"- {title} — {href}\n  {body}")
        return "\n".join(lines)
    except Exception as e:
        return f"Erreur recherche: {e}"