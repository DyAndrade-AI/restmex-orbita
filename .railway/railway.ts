import { defineRailway, github, project, service, volume } from "railway/iac";
// This repository manages only its own resources in the environment. Other
// repositories export their own partial name.
// See https://docs.railway.com/infrastructure-as-code#multi-repo-projects
export const partial = "restmex-orbita";
export default defineRailway(() => {
  // Juicios del laboratorio y pools congelados: sobreviven a cada deploy.
  const labels = volume("orbita-labels", { sizeMB: 500 });

  const restmexOrbita = service("restmex-orbita", {
    source: github("DyAndrade-AI/restmex-orbita", { branch: "main", checkSuites: false }),
    build: { buildEnvironment: "V3", builder: "DOCKERFILE", dockerfilePath: "docker/api.Dockerfile" },
    healthcheck: "/health",
    healthcheckTimeout: 120,
    replicas: { "iad": 1 },
    volumeMounts: { "/app/eval/labels": labels },
  });

  // Web estática (Astro + nginx); nginx reenvía /api y /health a la API por la red privada.
  const web = service("web", {
    source: github("DyAndrade-AI/restmex-orbita", { branch: "main", rootDirectory: "web", checkSuites: false }),
    build: { buildEnvironment: "V3", builder: "DOCKERFILE", dockerfilePath: "Dockerfile" },
    replicas: { "iad": 1 },
  });

  return project("authentic-optimism", {
    variables: { managed: false },
    resources: [labels, restmexOrbita, web],
  });
});
