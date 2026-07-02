# LLMOps 콘솔 프런트엔드 (web)

Vite + React + TypeScript + MUI 기반의 콘솔 프런트엔드. 기존 단일 파일 콘솔
(`llmops_core/console/static/index.html`)의 기능을 컴포넌트 기반으로 재구현한다.

## 스택
- Vite 5 / React 18 / TypeScript
- MUI 6 (`@mui/material`, `@mui/icons-material`, Emotion)
- react-router-dom 6 (App 라우팅)
- 테마: `docs/COLOR_GUIDELINE.md` 디자인 토큰 → `src/theme.ts` palette 매핑

## 실행
```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
```
- 백엔드 콘솔(FastAPI)을 포트 **4100**에서 먼저 실행:
  `uvicorn llmops_core.console.app:app --reload --port 4100`
- `/api/*` 요청은 vite dev 서버가 백엔드로 프록시한다(`vite.config.ts`).
  포트가 다르면 `VITE_API_TARGET`로 재정의(`.env.example` 참고).
- 우상단 열쇠 아이콘에서 **마스터 키**를 설정하면 이후 요청에 `X-Master-Key` 헤더로 전송된다.

## 빌드
```bash
npm run build      # tsc 타입체크 + vite 프로덕션 번들 → dist/
npm run preview
```

## 구조
```
src/
  main.tsx            # 진입점 (ThemeProvider, CssBaseline, Router)
  theme.ts            # 색상 토큰 → MUI palette
  api.ts              # fetch 래퍼 + X-Master-Key + ApiError
  nav.ts              # 네비게이션 그룹/항목 정의 (기존 NAV_GROUPS 이식)
  App.tsx             # 라우트 → 페이지 매핑 (미구현은 Placeholder)
  hooks/useApi.ts     # GET + 로딩/에러/reload
  components/         # Shell, MasterKeyDialog, PageHeader, StateViews
  pages/              # Home, Dashboard, Datasets, Releases, Placeholder
```

## 구현 현황
- ✅ 시작하기(Home), 대시보드·GPU(`/api/infra`,`/api/models`),
  데이터셋(`/api/data/datasets`,`/api/data/validate`), 승인·평가(`/api/releases`)
- ⏳ 나머지 네비게이션 항목은 Placeholder(준비중). `App.tsx`의 `IMPLEMENTED`에
  경로→컴포넌트를 추가하고 `nav.ts`의 `implemented: true`를 켜면 확장된다.
