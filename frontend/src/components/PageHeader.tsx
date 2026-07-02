import { Box, Typography } from "@mui/material";
import type { ReactNode } from "react";

export function PageHeader({ title, subtitle, action }: { title: string; subtitle?: string; action?: ReactNode }) {
  return (
    <Box sx={{ display: "flex", alignItems: "flex-start", mb: 3, gap: 2, flexWrap: "wrap" }}>
      <Box sx={{ flexGrow: 1 }}>
        <Typography variant="h1">{title}</Typography>
        {subtitle && (
          <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
            {subtitle}
          </Typography>
        )}
      </Box>
      {action}
    </Box>
  );
}
