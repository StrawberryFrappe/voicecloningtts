import { useEffect, useRef, useState } from "react";
import { api, type TrainingClip, type TrainStatus, type Voice } from "../api";
import { player, useEvents } from "../events";
import { errMsg, useToast } from "../toast";

const fmtTime = (s?: number | null) => {
  if (s == null) return "…";
  if (s < 60) return `${Math.round(s)} s`;
  const m = Math.floor(s / 60);
  return m < 60 ? `${m} min ${Math.round(s % 60)} s` : `${Math.floor(m / 60)} h ${m % 60} min`;
};

/** Per-voice Seed-VC fine-tuning: makes the real-time voice changer sound more like this person. */
export default function FineTunePanel({ voice, onChange }: { voice: Voice; onChange: () => void }) {
  const [clips, setClips] = useState<TrainingClip[]>([]);
  const [seconds, setSeconds] = useState(0);
  const [train, setTrain] = useState<TrainStatus | null>(null);
  const [steps, setSteps] = useState<number | null>(null);
  const [busy, setBusy] = useState<"" | "upload" | "start" | "compare-rec" | "compare">("");
  const fileRef = useRef<HTMLInputElement>(null);
  const toast = useToast();

  const load = async () => {
    const [c, t] = await Promise.all([api.trainingClips(voice.id), api.trainStatus()]);
    setClips(c.clips);
    setSeconds(c.seconds);
    setTrain(t);
    setSteps((s) => s ?? t.defaults.steps);
  };
  useEffect(() => {
    load().catch((e) => toast(errMsg(e), "error"));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [voice.id]);

  useEvents((e) => {
    if (e.type !== "vc_train") return;
    setTrain((prev) => ({ ...(prev as TrainStatus), ...(e as unknown as TrainStatus) }));
    if (e.voice_id === voice.id && (e.state === "done" || e.state === "error")) {
      if (e.state === "done") toast(`Fine-tune of "${voice.name}" finished`);
      else toast(`Fine-tune failed: ${e.error}`, "error");
      onChange();
    }
  });

  const mine = train?.voice_id === voice.id;
  const running = !!train && ["starting", "running", "cancelling"].includes(train.state);
  const otherRunning = running && !mine;
  const pct = mine && train?.max_steps ? Math.round(((train.step ?? 0) / train.max_steps) * 100) : 0;

  async function addAudio(f: File) {
    setBusy("upload");
    try {
      const up = await api.uploadAudio(f);
      const r = await api.addTraining(voice.id, up.upload_id);
      setClips(r.clips);
      setSeconds(r.seconds);
    } catch (e) {
      toast(errMsg(e), "error");
    } finally {
      setBusy("");
    }
  }

  async function start() {
    setBusy("start");
    try {
      setTrain(await api.startFinetune(voice.id, steps ?? train?.defaults.steps ?? 200));
    } catch (e) {
      toast(errMsg(e), "error");
    } finally {
      setBusy("");
    }
  }

  async function compare() {
    // Record ~5 s of the user talking, then convert it with the base and the fine-tuned model.
    try {
      setBusy("compare-rec");
      await api.recordStart("mic");
      toast("Recording 5 seconds: say something!");
      await new Promise((r) => setTimeout(r, 5000));
      const rec = await api.recordStopRaw();
      setBusy("compare");
      const r = await api.compareFinetune(voice.id, rec.recording_id);
      const toBlob = (b64: string) => new Blob([Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))], { type: "audio/wav" });
      toast("Playing: base model, then fine-tuned");
      await player.playBlob(toBlob(r.base_wav_b64));
      await player.playBlob(toBlob(r.finetuned_wav_b64));
    } catch (e) {
      toast(errMsg(e), "error");
    } finally {
      setBusy("");
    }
  }

  const minutes = seconds / 60;
  return (
    <div className="col" style={{ marginTop: 12 }}>
      <hr style={{ margin: "4px 0" }} />
      <div className="row between">
        <strong>Voice changer fine-tune</strong>
        {voice.vc_finetune
          ? <span className="badge accent">fine-tuned ✓ {voice.vc_finetune.steps} steps{voice.vc_finetune.seconds ? `, ${(voice.vc_finetune.seconds / 60).toFixed(1)} min audio` : ""}</span>
          : <span className="badge">not fine-tuned</span>}
      </div>
      <p className="small muted" style={{ margin: 0 }}>
        Trains the real-time voice changer on more audio of this person so it sounds closer to them (TTS is not affected).
        Aim for 1–5 minutes of clean speech; even 30 seconds helps. Long recordings are split at pauses automatically.
      </p>

      <div className="row small">
        <span>{clips.length} clip{clips.length === 1 ? "" : "s"} · {minutes >= 1 ? `${minutes.toFixed(1)} min` : `${Math.round(seconds)} s`}</span>
        <input ref={fileRef} type="file" style={{ display: "none" }}
          accept=".mp3,.mp4,.m4a,.wav,.ogg,.opus,.flac,.webm,.mkv,.mov,.aac,audio/*,video/*"
          onChange={(e) => { if (e.target.files?.[0]) void addAudio(e.target.files[0]); e.target.value = ""; }} />
        <button className="btn sm" disabled={!!busy || running} onClick={() => fileRef.current?.click()}>
          {busy === "upload" ? "Splitting…" : "+ Add training audio"}
        </button>
      </div>
      {clips.length > 1 && (
        <details>
          <summary className="small muted" style={{ cursor: "pointer" }}>Clips</summary>
          <div className="col" style={{ gap: 2, marginTop: 6, maxHeight: 160, overflow: "auto" }}>
            {clips.map((c) => (
              <div key={c.name} className="row small">
                <span className="grow">{c.name}</span><span className="muted">{c.seconds.toFixed(1)} s</span>
                {c.removable && !running && (
                  <button className="btn sm ghost" onClick={async () => {
                    const r = await api.deleteTraining(voice.id, c.name);
                    setClips(r.clips);
                    setSeconds(r.seconds);
                  }}>✕</button>
                )}
              </div>
            ))}
          </div>
        </details>
      )}

      {mine && running ? (
        <div className="col" style={{ gap: 6 }}>
          <div className="meter"><div style={{ width: `${pct}%`, background: "var(--accent)" }} /></div>
          <div className="row small">
            <span>{train?.message} step {train?.step ?? 0}/{train?.max_steps}</span>
            {train?.loss != null && <span className="muted">loss {train.loss}</span>}
            <span className="muted">ETA {fmtTime(train?.eta_s)}</span>
            <span className="grow" />
            <button className="btn sm danger" disabled={train?.state === "cancelling"} onClick={async () => setTrain(await api.cancelTrain())}>Cancel</button>
          </div>
          <span className="small muted">The GPU is reserved while training: voice changer and TTS pause until it finishes.</span>
        </div>
      ) : (
        <div className="row small">
          <label className="field" style={{ minWidth: 260 }}>
            <span className="row between"><span>Training steps</span><span>{steps}</span></span>
            <input type="range" min={train?.defaults.min_steps ?? 50} max={1500} step={50}
              value={steps ?? 200} onChange={(e) => setSteps(parseInt(e.target.value))} />
          </label>
          <button className="btn primary" disabled={!!busy || otherRunning} onClick={start}
            title={otherRunning ? "Another voice is being fine-tuned" : ""}>
            {busy === "start" ? "Starting…" : voice.vc_finetune ? "Re-train" : "Fine-tune"}
          </button>
          {mine && train?.state === "error" && <span className="error-text">{train.error}</span>}
        </div>
      )}

      {voice.vc_finetune && !running && (
        <div className="row small">
          <label className="check">
            <input type="checkbox" checked={voice.vc_use_finetune} onChange={async (e) => {
              await api.updateVoice(voice.id, { vc_use_finetune: e.target.checked });
              onChange();
            }} />Use fine-tuned model in the voice changer
          </label>
          <span className="grow" />
          <button className="btn sm" disabled={!!busy} onClick={compare} title="Record 5 s of yourself and hear base vs fine-tuned">
            {busy === "compare-rec" ? "Recording…" : busy === "compare" ? "Converting…" : "🎧 Compare"}
          </button>
          <button className="btn sm danger" onClick={async () => {
            if (!confirm("Delete the fine-tuned model for this voice?")) return;
            await api.deleteFinetune(voice.id);
            onChange();
          }}>Delete fine-tune</button>
        </div>
      )}
    </div>
  );
}
