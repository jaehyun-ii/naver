import type { SvgIconComponent } from "@mui/icons-material";
import HomeOutlined from "@mui/icons-material/HomeOutlined";
import RocketLaunchOutlined from "@mui/icons-material/RocketLaunchOutlined";
import ChatOutlined from "@mui/icons-material/ChatOutlined";
import DatasetOutlined from "@mui/icons-material/DatasetOutlined";
import AccountTreeOutlined from "@mui/icons-material/AccountTreeOutlined";
import TuneOutlined from "@mui/icons-material/TuneOutlined";
import EditNoteOutlined from "@mui/icons-material/EditNoteOutlined";
import MenuBookOutlined from "@mui/icons-material/MenuBookOutlined";
import ShieldOutlined from "@mui/icons-material/ShieldOutlined";
import AssessmentOutlined from "@mui/icons-material/AssessmentOutlined";
import SpeedOutlined from "@mui/icons-material/SpeedOutlined";
import TimelineOutlined from "@mui/icons-material/TimelineOutlined";
import CloudQueueOutlined from "@mui/icons-material/CloudQueueOutlined";
import DashboardOutlined from "@mui/icons-material/DashboardOutlined";
import ScienceOutlined from "@mui/icons-material/ScienceOutlined";
import FolderOutlined from "@mui/icons-material/FolderOutlined";
import RouteOutlined from "@mui/icons-material/RouteOutlined";
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

// 기존 콘솔(index.html)의 NAV_GROUPS 구조를 그대로 옮긴다.
export const NAV_GROUPS: NavGroup[] = [
  {
    title: "쉬운 시작",
    items: [
      { path: "/", label: "시작하기", icon: HomeOutlined },
      { path: "/wizard", label: "가이드 실행", icon: RocketLaunchOutlined, implemented: true },
      { path: "/play", label: "모델 사용(플레이그라운드)", icon: ChatOutlined, implemented: true },
    ],
  },
  {
    title: "고급 · 데이터/학습",
    items: [
      { path: "/data", label: "데이터셋", icon: DatasetOutlined, implemented: true },
      { path: "/pipe", label: "파이프라인(직접 실행)", icon: AccountTreeOutlined, implemented: true },
      { path: "/hpo", label: "파라미터 최적화", icon: TuneOutlined, implemented: true },
      { path: "/prompt", label: "프롬프트", icon: EditNoteOutlined, implemented: true },
    ],
  },
  {
    title: "고급 · 서빙/안전",
    items: [
      { path: "/rag", label: "RAG 지식베이스", icon: MenuBookOutlined, implemented: true },
      { path: "/safety", label: "가드레일·안전", icon: ShieldOutlined, implemented: true },
    ],
  },
  {
    title: "고급 · 운영/품질",
    items: [
      { path: "/evalv", label: "버전별 성능", icon: AssessmentOutlined, implemented: true },
      { path: "/bench", label: "벤치마크 평가", icon: SpeedOutlined, implemented: true },
      { path: "/drift", label: "드리프트 탐지", icon: TimelineOutlined, implemented: true },
      { path: "/serving", label: "서빙·카나리", icon: CloudQueueOutlined, implemented: true },
      { path: "/dash", label: "대시보드·GPU", icon: DashboardOutlined, implemented: true },
    ],
  },
  {
    title: "관측·자산 (통합 뷰)",
    items: [
      { path: "/mlflow", label: "실험·모델", icon: ScienceOutlined, implemented: true },
      { path: "/storage", label: "스토리지", icon: FolderOutlined, implemented: true },
      { path: "/traces", label: "트레이스", icon: RouteOutlined, implemented: true },
    ],
  },
  {
    title: "고급 · 거버넌스",
    items: [
      { path: "/rel", label: "승인·평가", icon: GavelOutlined, implemented: true },
      { path: "/keys", label: "테넌트 키", icon: VpnKeyOutlined, implemented: true },
      { path: "/sec", label: "보안·RBAC/감사", icon: SecurityOutlined, implemented: true },
    ],
  },
];

export const ALL_ITEMS: NavItem[] = NAV_GROUPS.flatMap((g) => g.items);
