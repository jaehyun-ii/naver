import { useEffect, useState } from "react";
import {
  Button, Dialog, DialogActions, DialogContent, DialogContentText,
  DialogTitle, TextField,
} from "@mui/material";
import { getMasterKey, setMasterKey } from "../api";

export function MasterKeyDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [value, setValue] = useState("");

  useEffect(() => {
    if (open) setValue(getMasterKey());
  }, [open]);

  const save = () => {
    setMasterKey(value.trim());
    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="xs">
      <DialogTitle>마스터 키 설정</DialogTitle>
      <DialogContent>
        <DialogContentText sx={{ mb: 2, fontSize: 14 }}>
          모든 API 요청에 <code>X-Master-Key</code> 헤더로 전송됩니다. 브라우저 localStorage에 저장됩니다.
        </DialogContentText>
        <TextField
          autoFocus fullWidth type="password" label="Master Key"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") save(); }}
        />
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>취소</Button>
        <Button variant="contained" onClick={save}>저장</Button>
      </DialogActions>
    </Dialog>
  );
}
