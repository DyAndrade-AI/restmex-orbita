FROM node:24-alpine AS builder

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

COPY astro.config.mjs tsconfig.json ./
COPY src ./src
COPY public ./public
RUN npm run build

FROM nginx:1.27-alpine

# La imagen de nginx sustituye ${API_UPSTREAM} al arrancar (solo variables definidas, no $uri).
ENV API_UPSTREAM=http://api:8765 \
    NGINX_ENTRYPOINT_LOCAL_RESOLVERS=1
COPY nginx.conf /etc/nginx/templates/default.conf.template
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=5 \
  CMD wget -q -O /dev/null http://127.0.0.1:8080/health || exit 1
