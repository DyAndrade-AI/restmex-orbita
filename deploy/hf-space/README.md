---
title: Órbita API
emoji: 🪐
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
short_description: API del recomendador Órbita (Pueblos Mágicos, Rest-Mex 2025)
---

API de Python de [Órbita](https://github.com/DyAndrade-AI/restmex-orbita). La web vive en Netlify
y llama a este Space. Prueba: `GET /health`.

Secrets del Space (Settings → Variables and secrets):

| Nombre | Valor |
|---|---|
| `ORBITA_CORS_ORIGIN` | URL del sitio en Netlify, p. ej. `https://orbita.netlify.app` |
| `SUPABASE_URL` | `https://<proyecto>.supabase.co` |
| `SUPABASE_KEY` | clave secreta de Supabase (nunca la publishable/anon) |
