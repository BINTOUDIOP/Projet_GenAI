FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .


ENV PORT 8080
EXPOSE 8080

# Commande pour démarrer l'application Chainlit
CMD ["sh", "-c", "chainlit run app_chainlit.py --host 0.0.0.0 --port $PORT"]