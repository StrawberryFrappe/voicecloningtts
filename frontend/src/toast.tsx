import { createContext, useCallback, useContext, useState, type ReactNode } from "react";

type Kind = "info" | "error" | "warn";
interface Toast { id: number; kind: Kind; text: string }

const Ctx = createContext<(text: string, kind?: Kind) => void>(() => {});

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const push = useCallback((text: string, kind: Kind = "info") => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t.slice(-4), { id, kind, text }]);
    window.setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), kind === "error" ? 7000 : 3500);
  }, []);
  return (
    <Ctx.Provider value={push}>
      {children}
      <div className="toasts">
        {toasts.map((t) => (
          <div key={t.id} className={`toast ${t.kind}`} onClick={() => setToasts((x) => x.filter((y) => y.id !== t.id))}>
            {t.text}
          </div>
        ))}
      </div>
    </Ctx.Provider>
  );
}

export const useToast = () => useContext(Ctx);

export function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
