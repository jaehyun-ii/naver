import { useEffect } from "react";

/** on 이 true 인 동안 fn 을 즉시 1회 + ms 간격으로 반복 호출한다. */
export function useAutoRefresh(fn: () => void, ms: number, on: boolean) {
  useEffect(() => {
    if (!on) return;
    fn();
    const t = setInterval(fn, ms);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [on, ms]);
}
