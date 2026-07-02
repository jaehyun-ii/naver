import { Card, CardContent, Typography } from "@mui/material";
import ConstructionIcon from "@mui/icons-material/Construction";
import { PageHeader } from "../components/PageHeader";

export default function Placeholder({ title }: { title: string }) {
  return (
    <>
      <PageHeader title={title} subtitle="이 화면은 아직 준비 중입니다." />
      <Card>
        <CardContent sx={{ display: "flex", alignItems: "center", gap: 1.5, py: 4, color: "text.secondary" }}>
          <ConstructionIcon />
          <Typography variant="body2">
            해당 콘솔 기능은 아직 이 프런트엔드로 이관되지 않았습니다.
          </Typography>
        </CardContent>
      </Card>
    </>
  );
}
