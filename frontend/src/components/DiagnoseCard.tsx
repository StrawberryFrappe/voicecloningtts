import { useEffect, useState } from "react";
import { api, type DoctorCheck, type DoctorReport } from "../api";
import { useEvents } from "../events";
import { errMsg, useToast } from "../toast";

const ICON: Record<string, string> = { ok: "✓", warn: "⚠", fail: "✗", skip: "–" };
const COLOR: Record<string, string> = { ok: "var(--good)", warn: "var(--warn)", fail: "var(--bad)", skip: "var(--muted)" };

/** One-click self-check; produces a report to paste when asking for help. */
export default function DiagnoseCard() {
  const [running, setRunning] = useState<"" | "quick" | "deep">("");
  const [live, setLive] = useState<DoctorCheck[]>([]);
  const [report, setReport] = useState<DoctorReport | null>(null);
  const toast = useToast();

  useEffect(() => {
    api.doctorLast().then((r) => r.report && setReport(r.report)).catch(() => {});
  }, []);

  useEvents((e) => {
    if (e.type === "doctor") setLive((l) => [...l, e.check as DoctorCheck]);
  });

  async function run(deep: boolean) {
    setRunning(deep ? "deep" : "quick");
    setLive([]);
    try {
      setReport(await api.doctor(deep));
    } catch (e) {
      toast(errMsg(e), "error");
    } finally {
      setRunning("");
    }
  }

  const checks = running ? live : report?.checks ?? [];
  const problems = checks.filter((c) => c.status === "fail" || c.status === "warn");

  return (
    <div className="card">
      <div className="row between">
        <h2 style={{ margin: 0 }}>Diagnose</h2>
        {report && !running && (
          <span className="small muted">
            last run {report.created_at}: {report.summary.ok} ok · {report.summary.warn} warnings · {report.summary.fail} failed
          </span>
        )}
      </div>
      <p className="small muted">
        Checks every part of the install (GPU in each environment, audio devices and VB-Cable, ffmpeg, models,
        network, keys). <b>Deep check</b> also loads the models (downloading them the first time), speaks a test
        sentence, measures the voice changer on this GPU, transcribes, and pings your LLM: it can take several minutes.
      </p>
      <div className="row">
        <button className="btn primary" disabled={!!running} onClick={() => run(false)}>
          {running === "quick" ? "Checking…" : "Quick check"}
        </button>
        <button className="btn" disabled={!!running} onClick={() => run(true)}>
          {running === "deep" ? "Running deep check…" : "Deep check"}
        </button>
        <span className="grow" />
        {report && !running && (
          <>
            <button className="btn sm" onClick={async () => {
              try {
                await navigator.clipboard.writeText(report.text);
                toast("Report copied: paste it when asking for help");
              } catch {
                toast("Couldn't access the clipboard; use Save report", "warn");
              }
            }}>Copy report</button>
            <button className="btn sm" onClick={() => {
              const url = URL.createObjectURL(new Blob([report.text], { type: "text/plain" }));
              const a = document.createElement("a");
              a.href = url;
              a.download = `vctts-diagnose-${report.created_at.replace(/[: ]/g, "-")}.txt`;
              a.click();
              URL.revokeObjectURL(url);
            }}>Save report</button>
          </>
        )}
      </div>

      {checks.length > 0 && (
        <div className="col" style={{ gap: 4, marginTop: 12 }}>
          {checks.map((c, i) => (
            <div key={`${c.id}-${i}`} className="row small" style={{ alignItems: "flex-start", flexWrap: "nowrap" }}>
              <span style={{ color: COLOR[c.status], width: 16, fontWeight: 700 }}>{ICON[c.status]}</span>
              <span style={{ width: 230, flexShrink: 0 }}>{c.title}</span>
              <span className="grow" style={{ minWidth: 0 }}>
                <span className={c.status === "fail" ? "error-text" : c.status === "warn" ? "warn-text" : "muted"}>{c.detail}</span>
                {c.fix && c.status !== "ok" && <div className="muted">→ {c.fix}</div>}
              </span>
            </div>
          ))}
          {running && <div className="small muted">…</div>}
        </div>
      )}
      {report && !running && problems.length === 0 && (
        <p className="small" style={{ color: "var(--good)" }}>Everything looks good.</p>
      )}
      {report?.saved_to && !running && <p className="small muted">Saved to <code>{report.saved_to}</code></p>}
    </div>
  );
}
