import { defineConfig } from "astro/config";

const api = process.env.ORBITA_API_URL ?? "http://127.0.0.1:8765";

export default defineConfig({
  server: {
    host: "127.0.0.1",
    port: 4321,
  },
  // El proxy es de Vite: `server.proxy` de Astro se ignora y el dev server devolvía 404 en /api.
  vite: {
    server: {
      proxy: {
        "/api": api,
        "/health": api,
      },
    },
  },
});
