// Base de la API. Vacía = mismo origen (proxy de Vite en dev, nginx en Docker). En un despliegue
// estático (Netlify) se define PUBLIC_API_URL con la URL pública de la API de Python.
export const API = String(import.meta.env.PUBLIC_API_URL ?? "").replace(/\/$/, "");
export const api = (path: string) => `${API}${path}`;
