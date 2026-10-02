import { useEffect, useState } from "react";
import { api, type Preset, type VCSettings, type VCStatus, type Voice } from "../api";
import { useEvents } from "../events";
import { errMsg, useToast } from "../toast";

const SLIDERS: { key: keyof VCSettings; label: string; min: number; max: number; step: number; unit: string; help: string }[] = [
  { key: "block_time", label: "Block size", min: 0.08, max: 0.6, step: 0.01, unit: "s", help: "Latency ≈ 2 × block. Must stay above the inference time." },
  { key: "diffusion_steps", label: "Quality (diffusion steps)", min: 1, max: 20, step: 1, unit: "", help: "4–10 is typical for real time. Higher = better but slower." },
  { key: "max_prompt_length", label: "Voice reference used", min: 1, max: 10, step: 0.5, unit: "s", help: "Longer = closer to the voice, slower." },
  { key: "inference_cfg_rate", label: "CFG rate", min: 0, max: 1, step: 0.05, unit: "", help: "0 is ~1.5× faster with a subtle quality change." },
  { key: "extra_time_ce", label: "Context (left)", min: 0.5, max: 5, step: 0.1, unit: "s", help: "More history = more stable, slower." },
  { key: "extra_time_right", label: "Look-ahead", min: 0.02, max: 0.5, step: 0.01, unit: "s", help: "Adds directly to latency." },
  { key: "gate_db", label: "Silence gate", min: -80, max: -20, step: 1, unit: "dB", help: "Below this your mic counts as silent (saves GPU)." },
];

export default function VoiceChangerCard({ voices, audioRunning, hasMic }: { voices: Voice[]; audioRunning: boolean; hasMic: boolean }) {
  const [st, setSt] = useState<VCStatus | null>(null);
  const [draft, setDraft] = useState<VCSettings | null>(null);
  const [voiceId, setVoiceId] = useState<string>("");
  const [presets, setPresets] = useState<{ presets: Record<string, Preset>; recommended: string } | null>(null);
  const [tuning, setTuning] = useState(false);
  const [tuneResult, setTuneResult] = useState<string | null>(null);
  const toast = useToast();

  useEffect(() => {
    api.vcStatus().then((s) => {
      setSt(s);
      setDraft(s.settings);
      setVoiceId(s.voice_id || s.saved_voice_id || voices[0]?.id || "");
    }).catch(() => {});
    api.vcPresets().then(setPresets).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEvents((e) => {
    if (e.type === "vc") setSt((prev) => ({ ...(prev as VCStatus), ...(e as unknown as VCStatus) }));
  });

  async function applyPreset(name: string) {
    const p = presets?.presets[name];
    if (!p) return;
    setDraft(p.settings);
    await apply(p.settings);
  }

  async function autoTune() {
    setTuning(true);
    setTuneResult(null);
    try {
      const r = await api.vcAutotune(voiceId || null);
      setSt(r.status);
      setDraft(r.status.settings);
      const label = (n: string) => presets?.presets[n]?.label ?? n;
      const parts = r.tried.map((t) => `${label(t.preset)} ${Math.round(t.load * 100)}%`);
      setTuneResult(r.ok
        ? `Picked ${label(r.preset)} (${parts.join(", ")} load).`
        : `Even the lightest preset can't keep up on this GPU (${parts.join(", ")}). Expect glitches; close other GPU apps.`);
    } catch (e) {
      toast(errMsg(e), "error");
    } finally {
      setTuning(false);
    }
  }

  if (!st || !draft) return null;
  const live = st.state === "live";
  const busy = st.state === "loading";
  const overloaded = live && st.load > 0.95;
  const voice = voices.find((v) => v.id === voiceId);
  const willUseFinetune = !!voice?.vc_finetune && voice.vc_use_finetune;

  async function start() {
    try {
      setSt(await api.vcStart(voiceId || null));
    } catch (e) {
      toast(errMsg(e), "error");
    }
  }
  async function apply(next: VCSettings) {
    try {
      setSt(await api.vcSettings(next, voiceId || null));
    } catch (e) {
      toast(errMsg(e), "error");
    }
  }

  return (
    <div className="card">
      <div className="row between">
        <h2 style={{ margin: 0 }}>Real-time voice changer</h2>
        {!st.available ? <span className="pill bad">not installed</span>
          : live ? <span className={`pill ${overloaded ? "warn" : "good"}`}>{overloaded ? "live: GPU can't keep up" : "live"}</span>
          : busy ? <span className="pill warn">loading model…</span>
          : st.state === "error" ? <span className="pill bad">error</span>
          : <span className="pill">off</span>}
      </div>
      <p className="small muted">
        Converts your live microphone into the selected cloned voice (Seed-VC, zero-shot) and sends it to the
        virtual mic instead of your real voice. TTS replies still mix on top. Needs a GPU for real-time use.
      </p>
      {!st.available && <p className="small warn-text">{st.reason}</p>}
      {st.error && <p className="small error-text">{st.error}</p>}

      <div className="row">
        <label className="field" style={{ minWidth: 220 }}>Voice
          <select value={voiceId} disabled={busy} onChange={(e) => {
            setVoiceId(e.target.value);
            if (live) void api.vcSettings(draft, e.target.value).then(setSt).catch((err) => toast(errMsg(err), "error"));
          }}>
            {voices.map((v) => <option key={v.id} value={v.id}>{v.name}</option>)}
          </select>
        </label>
        <span className="grow" />
        {live || busy
          ? <button className="btn danger solid" onClick={async () => setSt(await api.vcStop())}>■ Stop voice changer</button>
          : <button className="btn primary" disabled={!st.available || !voiceId || !audioRunning || !hasMic} onClick={start}
              title={!audioRunning ? "Start the virtual mic first" : !hasMic ? "Choose your microphone first" : ""}>
              🎭 Start voice changer
            </button>}
        {st.model_loaded && !live && !busy && (
          <button className="btn sm ghost" onClick={async () => setSt(await api.vcUnload())} title="Free GPU memory">Unload model</button>
        )}
      </div>
      <div className="row small" style={{ marginTop: 10 }}>
        <span className="muted">Preset:</span>
        {presets && Object.entries(presets.presets).map(([name, p]) => (
          <button key={name} className="btn sm" disabled={busy} onClick={() => applyPreset(name)}
            title={`${p.settings.diffusion_steps} steps, ${p.settings.block_time}s blocks`}>
            {p.label} <span className="muted">~{p.latency_ms} ms</span>
            {presets.recommended === name && " ★"}
          </button>
        ))}
        <button className="btn sm primary" disabled={!st.available || live || busy || tuning || !voiceId} onClick={autoTune}
          title="Measures this GPU and picks the best preset that keeps up (voice changer must be stopped)">
          {tuning ? "Measuring…" : "⚡ Auto-tune"}
        </button>
      </div>
      {tuneResult && <p className="small muted">{tuneResult}</p>}
      <div className="row small muted" style={{ marginTop: 4 }}>
        {willUseFinetune && <span className="badge accent">uses fine-tuned model</span>}
        {st.precision && st.precision !== "auto" && <span className="badge">{st.precision}</span>}
      </div>

      {(!audioRunning || !hasMic) && st.available && (
        <p className="small muted">Pick your microphone and start the virtual mic above to enable the voice changer.</p>
      )}

      {live && (
        <div className="kv" style={{ marginTop: 10 }}>
          <div>Latency (algorithm)</div><div>~{st.latency_ms} ms + device</div>
          <div>Inference / block</div>
          <div className={overloaded ? "warn-text" : ""}>{st.infer_ms} ms / {st.block_ms} ms ({Math.round(st.load * 100)}% load)</div>
          {(st.dropped_blocks > 0 || st.underruns > 0) && (<><div>Glitches</div><div className="warn-text">{st.dropped_blocks} dropped, {st.underruns} underruns. Raise block size or lower quality.</div></>)}
        </div>
      )}

      <details style={{ marginTop: 12 }}>
        <summary className="small muted" style={{ cursor: "pointer" }}>Tuning</summary>
        <div className="grid2" style={{ marginTop: 10 }}>
          {SLIDERS.map((s) => (
            <label key={s.key} className="field" title={s.help}>
              <span className="row between"><span>{s.label}</span><span>{draft[s.key]}{s.unit && ` ${s.unit}`}</span></span>
              <input type="range" min={s.min} max={s.max} step={s.step} value={draft[s.key]}
                onChange={(e) => setDraft({ ...draft, [s.key]: parseFloat(e.target.value) })}
                onPointerUp={() => apply(draft)} onKeyUp={() => apply(draft)} />
              <span className="small">{s.help}</span>
            </label>
          ))}
        </div>
      </details>
    </div>
  );
}
