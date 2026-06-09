# Frontend Web (Bônus)

## O que fazer
Criar uma interface web simples em HTML + JS que consome a API e permite testar as previsões visualmente.

Funcionalidades:
- Formulário com campos do cliente (tenure, MonthlyCharges, Contract, etc.)
- Botão "Prever Churn" que chama POST /predict e mostra resultado
- Seção de comparação que chama GET /compare e exibe lado a lado
- Seção de explicação que chama POST /explain e mostra o caminho da árvore
- Seção de cluster que chama POST /cluster e mostra o perfil

## Como fazer
- Criar `frontend/index.html` com HTML + CSS + JS puro (sem framework, sem build step)
- Usar `fetch()` para chamar a API
- Servir o frontend via FastAPI usando `StaticFiles` ou abrir direto no navegador
- Habilitar CORS na API com `CORSMiddleware` para permitir chamadas do frontend
- Manter o design simples e funcional — foco em mostrar que funciona

---
✅ Concluído
