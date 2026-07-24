import type { SvgIconComponent } from "@mui/icons-material";
import DashboardOutlined from "@mui/icons-material/DashboardOutlined";
import RocketLaunchOutlined from "@mui/icons-material/RocketLaunchOutlined";
import ChatOutlined from "@mui/icons-material/ChatOutlined";
import DatasetOutlined from "@mui/icons-material/DatasetOutlined";
import DescriptionOutlined from "@mui/icons-material/DescriptionOutlined";
import AccountTreeOutlined from "@mui/icons-material/AccountTreeOutlined";
import TuneOutlined from "@mui/icons-material/TuneOutlined";
import EditNoteOutlined from "@mui/icons-material/EditNoteOutlined";
import MenuBookOutlined from "@mui/icons-material/MenuBookOutlined";
import ShieldOutlined from "@mui/icons-material/ShieldOutlined";
import AssessmentOutlined from "@mui/icons-material/AssessmentOutlined";
import SpeedOutlined from "@mui/icons-material/SpeedOutlined";
import TimelineOutlined from "@mui/icons-material/TimelineOutlined";
import CloudQueueOutlined from "@mui/icons-material/CloudQueueOutlined";
import MemoryOutlined from "@mui/icons-material/Memory";
import ScienceOutlined from "@mui/icons-material/ScienceOutlined";
import FolderOutlined from "@mui/icons-material/FolderOutlined";
import RouteOutlined from "@mui/icons-material/RouteOutlined";
import FactCheckOutlined from "@mui/icons-material/FactCheckOutlined";
import GavelOutlined from "@mui/icons-material/GavelOutlined";
import VpnKeyOutlined from "@mui/icons-material/VpnKeyOutlined";
import SecurityOutlined from "@mui/icons-material/SecurityOutlined";

export interface NavItem {
  /** 라우트 경로 (home 은 "/") */
  path: string;
  label: string;
  icon: SvgIconComponent;
  /** 화면 구현 여부 — false 면 준비중 플레이스홀더로 렌더 */
  implemented?: boolean;
}

export interface NavGroup {
  title: string;
  items: NavItem[];
}

// LLMOps 라이프사이클(TNAI · Vertex AI · MLflow 종합):
// 개요 → 데이터 관리 → 모델 학습·평가 → 배포·서빙 → 모니터링·관측 → 안전·거버넌스
export const NAV_GROUPS: NavGroup[] = [
  {
    title: "개요",
    items: [
      { path: "/", label: "대시보드", icon: DashboardOutlined },
      { path: "/wizard", label: "가이드 실행", icon: RocketLaunchOutlined, implemented: true },
      { path: "/play", label: "플레이그라운드", icon: ChatOutlined, implemented: true },
    ],
  },
  {
    title: "데이터 관리",
    items: [
      { path: "/documents", label: "문서 관리", icon: DescriptionOutlined, implemented: true },
      { path: "/data", label: "데이터셋", icon: DatasetOutlined, implemented: true },
      { path: "/rag", label: "RAG 지식베이스", icon: MenuBookOutlined, implemented: true },
      { path: "/review", label: "청킹 리뷰", icon: FactCheckOutlined, implemented: true },
    ],
  },
  {
    title: "모델 학습 · 평가",
    items: [
      { path: "/pipe", label: "학습 파이프라인", icon: AccountTreeOutlined, implemented: true },
      { path: "/hpo", label: "하이퍼파라미터 최적화", icon: TuneOutlined, implemented: true },
      { path: "/evalv", label: "성능 비교", icon: AssessmentOutlined, implemented: true },
      { path: "/bench", label: "벤치마크", icon: SpeedOutlined, implemented: true },
      { path: "/prompt", label: "프롬프트 템플릿", icon: EditNoteOutlined, implemented: true },
    ],
  },
  {
    title: "배포 · 서빙",
    items: [
      { path: "/serving", label: "서빙 · 카나리", icon: CloudQueueOutlined, implemented: true },
      { path: "/mlflow", label: "모델 카탈로그", icon: ScienceOutlined, implemented: true },
      { path: "/rel", label: "승인 · 평가", icon: GavelOutlined, implemented: true },
    ],
  },
  {
    title: "모니터링 · 관측",
    items: [
      { path: "/drift", label: "드리프트 탐지", icon: TimelineOutlined, implemented: true },
      { path: "/traces", label: "트레이스", icon: RouteOutlined, implemented: true },
      { path: "/dash", label: "인프라 · GPU", icon: MemoryOutlined, implemented: true },
      { path: "/storage", label: "스토리지", icon: FolderOutlined, implemented: true },
    ],
  },
  {
    title: "안전 · 거버넌스",
    items: [
      { path: "/safety", label: "가드레일 · 안전", icon: ShieldOutlined, implemented: true },
      { path: "/keys", label: "테넌트 키", icon: VpnKeyOutlined, implemented: true },
      { path: "/sec", label: "보안 · RBAC/감사", icon: SecurityOutlined, implemented: true },
    ],
  },
];

export const ALL_ITEMS: NavItem[] = NAV_GROUPS.flatMap((g) => g.items);
