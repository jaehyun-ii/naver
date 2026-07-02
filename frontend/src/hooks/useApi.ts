import { useCallback, useEffect, useState } from "react";
import { api } from "../api";

interface State<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

/** GET 요청 + 로딩/에러/재조회(reload)를 관리하는 훅. */
export function useApi<T>(path: string, deps: unknown[] = []) {
  const [state, setState] = useState<State<T>>({ data: null, loading: true, error: null });

  const reload = useCallback(
    (signal?: AbortSignal) => {
      setState((s) => ({ ...s, loading: true, error: null }));
      return api<T>(path, { signal })
        .then((data) => setState({ data, loading: false, error: null }))
        .catch((e: unknown) => {
          if (e instanceof DOMException && e.name === "AbortError") return;
          setState({ data: null, loading: false, error: (e as Error).message });
        });
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [path, ...deps],
  );

  useEffect(() => {
    const ctrl = new AbortController();
    reload(ctrl.signal);
    return () => ctrl.abort();
  }, [reload]);

  return { ...state, reload };
}
