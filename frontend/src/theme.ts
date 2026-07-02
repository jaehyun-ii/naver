import { createTheme } from "@mui/material/styles";
import type { Shadows } from "@mui/material/styles";

// docs/COLOR_GUIDELINE.md 의 디자인 토큰을 MUI palette로 매핑한다.
// raw HEX는 여기(테마 정의)에서만 사용하고, 컴포넌트에서는 theme.palette를 통해 참조한다.
export const tokens = {
  primary: {
    5: "#EFF5FF", 10: "#D3E1FB", 50: "#246BEB", 60: "#1D56BC", 70: "#16408D",
  },
  secondary: {
    5: "#EDF1F5", 50: "#003675", 60: "#002B5E", 70: "#002046",
  },
  gray: {
    0: "#FFFFFF", 5: "#F8F8F8", 10: "#F0F0F0", 20: "#E4E4E4", 30: "#D8D8D8",
    40: "#C6C6C6", 50: "#8E8E8E", 60: "#717171", 70: "#555555", 80: "#2D2D2D",
    90: "#1D1D1D",
  },
  danger: { 5: "#FEECF0", 20: "#F799B1", 50: "#EB003B", 60: "#D50136", 70: "#8D0023" },
  warning: { 5: "#FFF8E9", 20: "#FFE2A7", 50: "#FFB724", 60: "#98690A" },
  success: { 5: "#EEF7F0", 20: "#B2DCBB", 50: "#008A1E", 60: "#006E18" },
  info: { 5: "#E9F0FF", 20: "#A9C3FF", 50: "#2768FF", 60: "#1F53CC" },
  // 토스풍 서피스: 쿨한 화이트 캔버스 + 아주 옅은 라인. (브랜드색과 별개의 중립 표면 토큰)
  surface: { canvas: "#F4F6F8", raised: "#FFFFFF", line: "#EAEDF1" },
} as const;

// 토스 스타일 소프트 섀도우 — 낮은 알파의 쿨한 슬레이트 톤으로 층을 은은하게 구분한다.
// 하드 보더 대신 그림자로 카드/팝오버의 깊이를 표현한다.
const SLATE = "23,27,33"; // #171B21
const softShadows = Array.from({ length: 25 }, (_, i) => {
  if (i === 0) return "none";
  const y = Math.max(1, Math.round(i * 0.9));
  const blur = 6 + i * 3;
  const a2 = (0.05 + Math.min(i, 10) * 0.006).toFixed(3);
  return `0 1px 2px rgba(${SLATE},0.04), 0 ${y}px ${blur}px rgba(${SLATE},${a2})`;
}) as unknown as Shadows;

// 카드/패널에 쓰는 은은한 기본 그림자(hover 시 살짝 떠오름).
const CARD_SHADOW = `0 1px 2px rgba(${SLATE},0.04), 0 4px 16px rgba(${SLATE},0.06)`;
const CARD_SHADOW_HOVER = `0 2px 4px rgba(${SLATE},0.05), 0 10px 28px rgba(${SLATE},0.10)`;

export const theme = createTheme({
  palette: {
    primary: {
      light: tokens.primary[10],
      main: tokens.primary[50],
      dark: tokens.primary[70],
      contrastText: tokens.gray[0],
    },
    secondary: {
      light: tokens.secondary[5],
      main: tokens.secondary[50],
      dark: tokens.secondary[70],
      contrastText: tokens.gray[0],
    },
    error: { light: tokens.danger[5], main: tokens.danger[50], dark: tokens.danger[70] },
    warning: { light: tokens.warning[5], main: tokens.warning[50], dark: tokens.warning[60] },
    success: { light: tokens.success[5], main: tokens.success[50], dark: tokens.success[60] },
    info: { light: tokens.info[5], main: tokens.info[50], dark: tokens.info[60] },
    grey: {
      50: tokens.gray[5], 100: tokens.gray[10], 200: tokens.gray[20], 300: tokens.gray[30],
      400: tokens.gray[40], 500: tokens.gray[50], 600: tokens.gray[60], 700: tokens.gray[70],
      800: tokens.gray[80], 900: tokens.gray[90],
    },
    background: { default: tokens.surface.canvas, paper: tokens.surface.raised },
    text: {
      primary: tokens.gray[90],
      secondary: tokens.gray[70],
      disabled: tokens.gray[50],
    },
    // 하드 보더 대신 그림자를 쓰므로 라인 색은 아주 옅게.
    divider: tokens.surface.line,
    action: {
      hover: "rgba(23,27,33,0.04)",
      selected: tokens.primary[5],
    },
  },
  // 토스풍의 넉넉하게 둥근 모서리.
  shape: { borderRadius: 12 },
  shadows: softShadows,
  typography: {
    fontFamily: [
      "Pretendard", "Pretendard Variable", "Roboto", "system-ui", "-apple-system",
      "Segoe UI", "Apple SD Gothic Neo", "Noto Sans KR", "sans-serif",
    ].join(","),
    // 크고 굵은 헤드라인 + 타이트한 자간(토스 특유의 또렷한 위계).
    h1: { fontSize: "1.9rem", fontWeight: 800, letterSpacing: "-0.02em", lineHeight: 1.25 },
    h2: { fontSize: "1.45rem", fontWeight: 800, letterSpacing: "-0.015em", lineHeight: 1.3 },
    h3: { fontSize: "1.15rem", fontWeight: 700, letterSpacing: "-0.01em" },
    subtitle1: { fontWeight: 700 },
    subtitle2: { fontWeight: 700 },
    body1: { lineHeight: 1.6 },
    body2: { lineHeight: 1.6 },
    button: { fontWeight: 700, letterSpacing: 0 },
  },
  components: {
    MuiButton: {
      defaultProps: { disableElevation: true },
      styleOverrides: {
        root: { borderRadius: 12, textTransform: "none", paddingInline: 18 },
        sizeMedium: { minHeight: 42 },
        sizeLarge: { minHeight: 52, fontSize: "1rem", borderRadius: 14, paddingInline: 24 },
        containedPrimary: {
          boxShadow: "none",
          "&:hover": { boxShadow: "none" },
        },
      },
    },
    MuiCard: {
      defaultProps: { elevation: 0 },
      styleOverrides: {
        // 보더 제거 → 소프트 섀도우. 넉넉한 라운드.
        root: {
          border: "none",
          borderRadius: 20,
          boxShadow: CARD_SHADOW,
          transition: "box-shadow .2s ease, transform .2s ease",
        },
      },
    },
    MuiCardContent: {
      styleOverrides: {
        root: { padding: 24, "&:last-child": { paddingBottom: 24 } },
      },
    },
    MuiCardActionArea: {
      styleOverrides: {
        root: {
          borderRadius: 20,
          "&:hover": { boxShadow: CARD_SHADOW_HOVER },
          transition: "box-shadow .2s ease",
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        // 팝오버/메뉴 등도 부드러운 그림자로.
        rounded: { borderRadius: 16 },
        outlined: ({ theme }) => ({ borderColor: theme.palette.divider }),
      },
    },
    MuiOutlinedInput: {
      styleOverrides: {
        root: { borderRadius: 12 },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: { borderRadius: 8, fontWeight: 600 },
        sizeSmall: { borderRadius: 8 },
      },
    },
    MuiTableCell: {
      styleOverrides: {
        root: ({ theme }) => ({
          borderColor: theme.palette.divider,
          paddingTop: 12,
          paddingBottom: 12,
        }),
        head: ({ theme }) => ({
          fontWeight: 700,
          color: theme.palette.text.secondary,
          background: "transparent",
        }),
      },
    },
    MuiTooltip: {
      styleOverrides: {
        tooltip: { borderRadius: 8, fontSize: 12, padding: "6px 10px" },
      },
    },
    MuiAlert: {
      styleOverrides: {
        root: { borderRadius: 12 },
      },
    },
    MuiLinearProgress: {
      styleOverrides: {
        root: { borderRadius: 999, height: 8, backgroundColor: tokens.surface.line },
        bar: { borderRadius: 999 },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: { boxShadow: "none" },
      },
    },
  },
});
