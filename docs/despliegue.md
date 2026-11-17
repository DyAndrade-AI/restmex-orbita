# Despliegue: Netlify (web) + Hugging Face Spaces (API) + Supabase (juicios)

```
Navegador ──► Netlify (web estática, videos)
    │
    └──► Hugging Face Space (API de Python, modo híbrido con E5) ──► Supabase (juicios del laboratorio)
```

La clave de Supabase solo la conoce la API: el navegador nunca habla con Supabase.

## 1. Supabase (≈3 min)

1. Crea un proyecto gratuito en [supabase.com](https://supabase.com).
2. **SQL Editor → New query**, pega y ejecuta:

   ```sql
   create table public.lab_judgments (
     id bigint generated always as identity primary key,
     created_at timestamptz not null default now(),
     record jsonb not null
   );
   -- Con RLS activo y sin políticas, las claves públicas no pueden leer ni escribir;
   -- solo la clave secreta de la API (que ignora RLS).
   alter table public.lab_judgments enable row level security;
   ```
3. **Project Settings → API Keys**: copia la URL del proyecto y la **secret key**
   (`sb_secret_…`; en proyectos antiguos, `service_role`). No uses la publishable/anon.

## 2. Hugging Face Space (≈10 min, la mayoría es el build)

1. [huggingface.co/new-space](https://huggingface.co/new-space): nombre `orbita-api`, SDK **Docker**
   (Blank), hardware **CPU basic (gratis)**, visibilidad pública.
2. Sube los dos archivos de `deploy/hf-space/` (`Dockerfile` y `README.md`) con *Files → Add file*.
3. **Settings → Variables and secrets → New secret**: `SUPABASE_URL`, `SUPABASE_KEY` y
   `ORBITA_CORS_ORIGIN` (pon la URL de Netlify cuando la tengas; puedes editarla después).
4. Espera a que diga *Running* y abre `https://<usuario>-orbita-api.hf.space/health`: debe decir
   `"mode": "híbrido"` (E5 tarda ~1 min en cargar tras arrancar; antes dice léxico).

Para actualizar la API tras un push a `main`: *Settings → Factory rebuild*.
El Space gratuito se duerme tras 48 h sin visitas; la primera petición lo despierta (~1–2 min).

## 2 bis. Railway (alternativa a Hugging Face, la que está en producción)

La API léxica (`docker/api.Dockerfile`, sin torch) corre en Railway; `railway.json` fija el
Dockerfile y el healthcheck `/health`. Railway inyecta `PORT` y la API lo respeta.

1. Railway → *New project → Deploy from GitHub repo* → `restmex-orbita` (auto-deploy en cada push
   a `main`).
2. **Variables**: `SUPABASE_URL`, `SUPABASE_KEY` y `ORBITA_CORS_ORIGIN` (URL de Netlify).
3. *Settings → Networking → Generate domain*. Comprueba `https://<dominio>/health` → `"mode": "léxico"`.
4. En Netlify cambia `PUBLIC_API_URL` a ese dominio y vuelve a desplegar.

No se duerme como el Space gratuito. El modo léxico empata con el híbrido en el holdout
(docs/experimentos_hybrid.md), así que no hace falta E5 en producción.

## 3. Netlify (≈3 min)

1. *Add new site → Import an existing project → GitHub* → `restmex-orbita`. La configuración sale
   de `netlify.toml` (base `web`, `npm run build`, publica `dist`).
2. **Site configuration → Environment variables**: `PUBLIC_API_URL` =
   `https://<usuario>-orbita-api.hf.space` (sin `/` final). No es secreta.
3. Deploy. Si cambias `PUBLIC_API_URL`, vuelve a desplegar: se incrusta al construir.
4. Copia la URL del sitio (`https://<nombre>.netlify.app`) al secret `ORBITA_CORS_ORIGIN` del Space
   y reinícialo.

## Comprobación

- La web carga y la consola de Órbita dice `MOTOR HÍBRIDO / EN LÍNEA`.
- Califica una consulta en *Ayúdanos a evaluar* → aparece una fila en `lab_judgments` en Supabase.
- Exportar todo: `https://<usuario>-orbita-api.hf.space/api/lab/export?format=csv`.
