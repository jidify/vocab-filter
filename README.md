# vocab-filter

## Backend LLM : Ollama ou CatGPT-Gateway

Les appels JSON de S3/S5 utilisent Ollama par défaut. Pour les envoyer à une
instance [CatGPT-Gateway](https://github.com/GautamVhavle/CatGPT-Gateway) déjà
lancée et authentifiée :

```powershell
uv run python run_pipeline.py --llm-backend catgpt `
  --llm-base-url http://localhost:8000/v1 `
  --catgpt-api-token dummy123 `
  --llm-model catgpt-browser
```

Pour un module lancé directement, utiliser `VOCAB_LLM_BACKEND=catgpt` avec
`CATGPT_BASE_URL`, `CATGPT_API_TOKEN`, `CATGPT_MODEL` et `CATGPT_TIMEOUT`.
Le gateway est un service distinct : ce projet ne le lance pas et ne gère pas
sa connexion à ChatGPT. Sans configuration, Ollama reste utilisé (`OLLAMA_URL`
et `OLLAMA_MODEL` permettent d'en changer l'adresse et le modèle).
