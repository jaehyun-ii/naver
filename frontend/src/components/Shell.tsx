import { useState, type ReactNode } from "react";
import { Link as RouterLink, useLocation } from "react-router-dom";
import {
  AppBar, Box, Collapse, Drawer, IconButton, List, ListItemButton, ListItemIcon,
  ListItemText, Toolbar, Typography, Tooltip,
} from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";
import KeyIcon from "@mui/icons-material/VpnKey";
import ExpandLess from "@mui/icons-material/ExpandLess";
import ExpandMore from "@mui/icons-material/ExpandMore";
import { NAV_GROUPS } from "../nav";
import { MasterKeyDialog } from "./MasterKeyDialog";

const DRAWER_WIDTH = 248;

export function Shell({ children }: { children: ReactNode }) {
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [keyOpen, setKeyOpen] = useState(false);
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>({});
  // 현재 페이지가 속한 그룹만 기본 펼침 — 나머지는 접어 사이드바를 짧게 유지
  const activeTitle = NAV_GROUPS.find((g) => g.items.some((i) => i.path === location.pathname))?.title;

  const drawer = (
    <Box role="navigation">
      <Toolbar sx={{ px: 2.5 }}>
        <Typography variant="h3" color="primary" fontWeight={700}>
          LLMOps 콘솔
        </Typography>
      </Toolbar>
      {NAV_GROUPS.map((group) => {
        const opened = group.title in openGroups ? openGroups[group.title] : group.title === activeTitle;
        return (
          <List key={group.title} dense sx={{ py: 0 }}>
            <ListItemButton
              onClick={() => setOpenGroups((o) => ({ ...o, [group.title]: !opened }))}
              sx={{ mx: 1, mt: 1, borderRadius: 2.5, py: 0.25 }}
              aria-expanded={opened}
            >
              <ListItemText
                primary={group.title}
                primaryTypographyProps={{
                  fontSize: 11, letterSpacing: ".5px", textTransform: "uppercase",
                  color: "text.secondary", fontWeight: 700,
                }}
              />
              {opened
                ? <ExpandLess fontSize="small" sx={{ color: "text.disabled" }} />
                : <ExpandMore fontSize="small" sx={{ color: "text.disabled" }} />}
            </ListItemButton>
            <Collapse in={opened} timeout="auto" unmountOnExit>
              {group.items.map((item) => {
                const selected = location.pathname === item.path;
                const Icon = item.icon;
                return (
                  <ListItemButton
                    key={item.path}
                    component={RouterLink}
                    to={item.path}
                    selected={selected}
                    onClick={() => setMobileOpen(false)}
                    sx={{
                      mx: 1, my: 0.25, borderRadius: 2.5,
                      "&.Mui-selected": {
                        bgcolor: "action.selected",
                        "&:hover": { bgcolor: "action.selected" },
                      },
                    }}
                  >
                    <ListItemIcon sx={{ minWidth: 36, color: selected ? "primary.main" : "text.secondary" }}>
                      <Icon fontSize="small" />
                    </ListItemIcon>
                    <ListItemText
                      primary={item.label}
                      primaryTypographyProps={{
                        fontSize: 13.5,
                        fontWeight: selected ? 600 : 400,
                        color: selected ? "primary.dark" : "text.primary",
                      }}
                    />
                  </ListItemButton>
                );
              })}
            </Collapse>
          </List>
        );
      })}
    </Box>
  );

  return (
    <Box sx={{ display: "flex", minHeight: "100vh" }}>
      <AppBar
        position="fixed"
        color="inherit"
        elevation={0}
        sx={{ zIndex: (t) => t.zIndex.drawer + 1, bgcolor: "background.paper" }}
      >
        <Toolbar>
          <IconButton
            edge="start" color="inherit" aria-label="메뉴 열기"
            onClick={() => setMobileOpen((v) => !v)}
            sx={{ mr: 2, display: { md: "none" } }}
          >
            <MenuIcon />
          </IconButton>
          <Box sx={{ flexGrow: 1 }} />
          <Tooltip title="마스터 키 설정">
            <IconButton color="primary" aria-label="마스터 키 설정" onClick={() => setKeyOpen(true)}>
              <KeyIcon />
            </IconButton>
          </Tooltip>
        </Toolbar>
      </AppBar>

      {/* 모바일 임시 Drawer */}
      <Drawer
        variant="temporary"
        open={mobileOpen}
        onClose={() => setMobileOpen(false)}
        ModalProps={{ keepMounted: true }}
        sx={{
          display: { xs: "block", md: "none" },
          "& .MuiDrawer-paper": { width: DRAWER_WIDTH, boxSizing: "border-box" },
        }}
      >
        {drawer}
      </Drawer>

      {/* 데스크톱 상시 Drawer */}
      <Drawer
        variant="permanent"
        sx={{
          display: { xs: "none", md: "block" },
          width: DRAWER_WIDTH, flexShrink: 0,
          "& .MuiDrawer-paper": { width: DRAWER_WIDTH, boxSizing: "border-box", border: "none", bgcolor: "background.paper" },
        }}
        open
      >
        {drawer}
      </Drawer>

      <Box component="main" sx={{ flexGrow: 1, p: { xs: 2, md: 4 }, width: { md: `calc(100% - ${DRAWER_WIDTH}px)` } }}>
        <Toolbar />
        {/* 모든 페이지 전체 폭(리뷰 페이지와 동일) — 가로 여백은 main 패딩(md:4)만 유지 */}
        <Box sx={{ maxWidth: "none", mx: "auto" }}>
          {children}
        </Box>
      </Box>

      <MasterKeyDialog open={keyOpen} onClose={() => setKeyOpen(false)} />
    </Box>
  );
}
