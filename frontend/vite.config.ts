import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 백엔드 콘솔(FastAPI)은 기본 포트 4100에서 동작한다.
// 개발 중 /api/* 요청을 백엔드로 프록시해 CORS/혼합 콘텐츠 문제를 피한다.
// 다른 포트를 쓰면 VITE_API_TARGET 환경변수로 재정의한다.
const API_TARGET = process.env.VITE_API_TARGET ?? "http://localhost:4100";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: API_TARGET, changeOrigin: true },
    },
  },
});
