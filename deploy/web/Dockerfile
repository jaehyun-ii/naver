# 프런트엔드(web) 이미지 — Vite+React 빌드 → nginx 정적 서빙 + /api 프록시.
# 멀티스테이지: 빌드는 컨테이너 내부에서 수행하므로 호스트 npm이 필요 없다(H100 제약 충족).
# build context = repo root (compose에서 지정). 산출물(dist)은 git에 커밋하지 않는다.

# ── 1) 빌드 스테이지 ──────────────────────────────────────────────
FROM node:24-alpine AS build
WORKDIR /app/frontend

# 의존성 레이어 캐시: 락파일 먼저 복사 후 설치
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# 소스 복사 후 타입체크 포함 빌드(vite build)
COPY frontend/ ./
RUN npm run build

# ── 2) 서빙 스테이지 ──────────────────────────────────────────────
FROM nginx:alpine AS serve
COPY deploy/web/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/frontend/dist /usr/share/nginx/html
EXPOSE 80
