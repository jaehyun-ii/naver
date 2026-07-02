import { Link as RouterLink } from "react-router-dom";
import { Box, Button, Card, CardActionArea, CardContent, Typography } from "@mui/material";
import ChatOutlined from "@mui/icons-material/ChatOutlined";
import DatasetOutlined from "@mui/icons-material/DatasetOutlined";
import DashboardOutlined from "@mui/icons-material/DashboardOutlined";
import GavelOutlined from "@mui/icons-material/GavelOutlined";
import type { SvgIconComponent } from "@mui/icons-material";
import { PageHeader } from "../components/PageHeader";

const SHORTCUTS: { path: string; label: string; desc: string; icon: SvgIconComponent }[] = [
  { path: "/play", label: "플레이그라운드", desc: "가상키로 배포된 모델과 대화", icon: ChatOutlined },
  { path: "/data", label: "데이터셋", desc: "JSONL 검증·빌드, 데이터 버전 관리", icon: DatasetOutlined },
  { path: "/dash", label: "대시보드·GPU", desc: "인프라·GPU·서비스 상태 한눈에", icon: DashboardOutlined },
  { path: "/rel", label: "승인·평가", desc: "릴리즈 승인/반려 거버넌스", icon: GavelOutlined },
];

export default function Home() {
  return (
    <>
      <PageHeader
        title="LLMOps 콘솔"
        subtitle="업로드 → 학습 → 평가 → 승인 → 배포까지 한 곳에서 관리합니다."
      />
      <Box
        sx={{ display: "grid", gap: 2, gridTemplateColumns: { xs: "1fr", sm: "1fr 1fr", md: "1fr 1fr 1fr" } }}
      >
        {SHORTCUTS.map((s) => {
          const Icon = s.icon;
          return (
            <Card key={s.path}>
              <CardActionArea component={RouterLink} to={s.path} sx={{ height: "100%" }}>
                <CardContent>
                  <Icon color="primary" sx={{ mb: 1 }} />
                  <Typography variant="h3" gutterBottom>{s.label}</Typography>
                  <Typography variant="body2" color="text.secondary">{s.desc}</Typography>
                </CardContent>
              </CardActionArea>
            </Card>
          );
        })}
      </Box>
      <Box sx={{ mt: 4 }}>
        <Button variant="contained" size="large" component={RouterLink} to="/data">
          데이터셋으로 시작하기
        </Button>
      </Box>
    </>
  );
}
