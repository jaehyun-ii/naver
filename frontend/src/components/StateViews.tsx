import { Alert, Box, CircularProgress, Typography } from "@mui/material";

export function Loading({ label = "불러오는 중…" }: { label?: string }) {
  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, py: 4, color: "text.secondary" }}>
      <CircularProgress size={20} />
      <Typography variant="body2">{label}</Typography>
    </Box>
  );
}

export function ErrorView({ message }: { message: string }) {
  return (
    <Alert severity="error" sx={{ my: 2 }}>
      {message}
    </Alert>
  );
}

export function EmptyView({ message }: { message: string }) {
  return (
    <Typography variant="body2" color="text.secondary" sx={{ py: 3 }}>
      {message}
    </Typography>
  );
}
