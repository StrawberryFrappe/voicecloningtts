import { useEffect, useRef, useState } from "react";
import { api, type Settings } from "../api";
import { useEvents } from "../events";
import { errMsg, useToast } from "../toast";
import LevelMeter from "./LevelMeter";

type Source = "mic" | "loopback";

/**
 * Record → transcribe → hand the text back for editing.
 * Uses the backend recorder (any mic, or WASAPI loopback of your headphones).
 * If the backend can't open the mic, falls back to recording in the browser.
 */
export default function RecordButton({ onText, disabled }: { onText: (t: string) => void; disabled?: boolean }) {
  const [source, setSource] = useState<Source>("mic");
  const [device, setDevice] = useState<string>("");
  const [state, setState] = useState<"idle" | "recording" | "transcribing">("idle");
  const [elapsed, setElapsed] = useState(0);
  const [level, setLevel] = useState(0);
  const browserRec = useRef<{ rec: MediaRecorder; chunks: Blob[]; stream: MediaStream } | null>(null);
  const started = useRef(0);
  const toast = useToast();

  useEffect(() => {
    api.settings().then((s: Settings) => {
      setSource(s.record_source || "mic");
      setDevice(s.record_device || "");
    }).catch(() => {});
  }, []);

  useEffect(() => {
    if (state !== "recording") return;
    const t = window.setInterval(() => setElapsed((Date.now() - started.current) / 1000), 200);
    return () => window.clearInterval(t);
  }, [state]);

  useEvents((e) => {
    if (e.type === "levels" && e.recorder) setLevel(e.recorder.level ?? 0);
  });

  async function start() {
    started.current = Date.now();
    setElapsed(0);
    try {
      await api.recordStart(source, device || null);
      setState("recording");
    } catch (err) {
      if (source === "mic" && navigator.mediaDevices?.getUserMedia) {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          const rec = new MediaRecorder(stream);
          const chunks: Blob[] = [];
          rec.ondataavailable = (ev) => chunks.push(ev.data);
          rec.start();
          browserRec.current = { rec, chunks, stream };
          setState("recording");
          toast("Recording in the browser (backend mic unavailable)", "warn");
          return;
        } catch {
          /* fall through */
        }
      }
      toast(errMsg(err), "error");
    }
  }

  async function stop() {
    setState("transcribing");
    try {
      let text = "";
      if (browserRec.current) {
        const { rec, chunks, stream } = browserRec.current;
        browserRec.current = null;
        await new Promise<void>((resolve) => {
          rec.onstop = () => resolve();
          rec.stop();
        });
        stream.getTracks().forEach((t) => t.stop());
        const res = await api.transcribeFile(new Blob(chunks, { type: rec.mimeType || "audio/webm" }));
        text = res.text;
      } else {
        const res = await api.recordStop(true);
        text = res.text ?? "";
      }
      if (text.trim()) onText(text.trim());
      else toast("No speech detected", "warn");
    } catch (err) {
      toast(errMsg(err), "error");
    } finally {
      setState("idle");
      setLevel(0);
    }
  }

  async function cancel() {
    if (browserRec.current) {
      browserRec.current.rec.stop();
      browserRec.current.stream.getTracks().forEach((t) => t.stop());
      browserRec.current = null;
    } else {
      await api.recordCancel().catch(() => {});
    }
    setState("idle");
  }

  return (
    <div className="rec-bar">
      <button
        className={`btn sm rec ${state === "recording" ? "active" : ""}`}
        disabled={disabled || state === "transcribing"}
        onClick={state === "recording" ? stop : start}
        title="Record a message, then edit the transcript before sending"
      >
        {state === "recording" ? "■ Stop" : state === "transcribing" ? "Transcribing…" : "● Record"}
      </button>
      {state === "idle" && (
        <div className="seg">
          <button className={source === "mic" ? "on" : ""} onClick={() => setSource("mic")}>Microphone</button>
          <button className={source === "loopback" ? "on" : ""} onClick={() => setSource("loopback")}>Headphones</button>
        </div>
      )}
      {state === "recording" && (
        <>
          <span>{elapsed.toFixed(1)}s</span>
          <div style={{ width: 140 }}><LevelMeter value={level} /></div>
          <button className="btn sm ghost" onClick={cancel}>Cancel</button>
        </>
      )}
    </div>
  );
}
