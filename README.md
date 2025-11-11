# Assistant Intelligent Multi-Compétences (RAG + Agents)

## Démarrage
```bash
python -m venv .venv && source .venv/bin/activate          # (Windows: .venv\Scripts\activate)
cp .env.example .env                                         # remplis tes clés si besoin
pip install -r requirements.txt

# 1) Mets tes fichiers dans ./data
python src/ingest.py

# 2) Lance l'interface
streamlit run src/interface_sl.py