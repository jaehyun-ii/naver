// 파이프라인/평가 단계 상태의 색상·한글 라벨 공용 매핑.

/** MUI palette 경로 (sx color 로 사용) */
export function stageColor(status: string): string {
  switch (status) {
    case "succeeded": return "success.main";
    case "failed": return "error.main";
    case "running":
    case "post-running": return "primary.main";
    case "waiting": return "warning.main";
    default: return "text.secondary"; // pending, skipped 등
  }
}

const STATUS_KO: Record<string, string> = {
  running: "진행 중",
  "post-running": "마무리 중",
  succeeded: "완료",
  failed: "실패",
  waiting: "대기(승인 필요)",
  pending: "대기",
  skipped: "건너뜀",
};

export function koStatus(status: string): string {
  return STATUS_KO[status] ?? status;
}

/** 파이프라인 stage name → 한글 라벨 */
export const STAGE_KO: Record<string, string> = {
  "data-load": "데이터 적재",
  "data-quality": "품질 검증",
  "data-build": "데이터 준비",
  finetune: "학습",
  merge: "병합",
  evaluate: "평가",
  register: "등록",
  deploy: "배포",
  canary: "카나리",
};
