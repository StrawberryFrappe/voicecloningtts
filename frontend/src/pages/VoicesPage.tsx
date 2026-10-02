import { useEffect, useRef, useState } from "react";
import { api, LANGUAGE_NAMES, type EngineInfo, type SettingSpec, type Voice } from "../api";
import { player } from "../events";
import { errMsg, useToast } from "../toast";
import Waveform from "../components/Waveform";
import FineTunePanel from "../components/FineTunePanel";

interface Upload { upload_id: string; filename: string; duration: number; peaks: number[] }

export default function VoicesPage({ onChange }: { onChange: () => void }) {
  const [voices, setVoices] = useState<Voice[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [engines, setEngines] = useState<EngineInfo[]>([]);
  const [editing, setEditing] = useState<string | null>(null);
  const toast = useToast();

  async function load() {
    const [v, e] = await Promise.all([api.voices(), api.engines()]);
    setVoices(v.voices);
    setActiveId(v.active_id);
    setEngines(e);
  }
  useEffect(() => {
    load().catch((e) => toast(errMsg(e), "error"));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const refresh = async () => {
    await load();
    onChange();
  };

  return (
    <div className="page page-narrow">
      <h1>Voices</h1>
      <p className="muted">Clone a voice from an mp3, mp4 (or any audio/video file) of the person speaking. Pick 10–30 seconds of
        clean speech with no music or other voices. Saved voices can be switched at any time and exported to share.</p>

      <NewVoice engines={engines} onCreated={refresh} />

      <div className="row between" style={{ margin: "18px 0 10px" }}>
        <h2 style={{ margin: 0 }}>Saved voices</h2>
        <ImportButton onImported={refresh} />
      </div>
      {voices.length === 0 && <div className="card muted">No voices yet.</div>}
      {voices.map((v) => (
        <VoiceCard
          key={v.id}
          voice={v}
          active={v.id === activeId}
          engines={engines}
          editing={editing === v.id}
          onEdit={() => setEditing(editing === v.id ? null : v.id)}
          onChange={refresh}
        />
      ))}

      <h2 style={{ marginTop: 22 }}>Engines</h2>
      <EnginesPanel engines={engines} onChange={load} />
    </div>
  );
}

function NewVoice({ engines, onCreated }: { engines: EngineInfo[]; onCreated: () => void }) {
  const [upload, setUpload] = useState<Upload | null>(null);
  const [busy, setBusy] = useState<"" | "upload" | "create">("");
  const [sel, setSel] = useState<[number, number]>([0, 0]);
  const [name, setName] = useState("");
  const [language, setLanguage] = useState("en");
  const [engine, setEngine] = useState("chatterbox");
  const [playhead, setPlayhead] = useState<number | null>(null);
  const audio = useRef<HTMLAudioElement | null>(null);
  const toast = useToast();

  const eng = engines.find((e) => e.id === engine);
  const langs = eng?.languages ?? { en: "English", es: "Spanish" };

  async function onFile(f: File) {
    setBusy("upload");
    try {
      const up = await api.uploadAudio(f);
      setUpload(up);
      setSel([0, Math.min(up.duration, 30)]);
      setName(f.name.replace(/\.[^.]+$/, ""));
    } catch (e) {
      toast(errMsg(e), "error");
    } finally {
      setBusy("");
    }
  }

  function playSelection() {
    if (!upload) return;
    audio.current?.pause();
    const a = new Audio(`/api/uploads/${upload.upload_id}/audio`);
    audio.current = a;
    a.currentTime = sel[0];
    const tick = () => {
      setPlayhead(a.currentTime);
      if (a.currentTime >= sel[1] || a.paused) {
        a.pause();
        setPlayhead(null);
        return;
      }
      requestAnimationFrame(tick);
    };
    void a.play().then(() => requestAnimationFrame(tick));
  }

  async function create() {
    if (!upload) return;
    setBusy("create");
    try {
      const res = await api.createVoice({
        upload_id: upload.upload_id, name: name || "New voice", language, engine,
        start: sel[0], end: sel[1] > sel[0] ? sel[1] : null,
      });
      res.warnings.forEach((w) => toast(w, "warn"));
      toast(`Voice "${res.voice.name}" saved`);
      setUpload(null);
      onCreated();
    } catch (e) {
      toast(errMsg(e), "error");
    } finally {
      setBusy("");
    }
  }

  const selLen = Math.max(0, sel[1] - sel[0]);
  return (
    <div className="card">
      <h2>New voice</h2>
      <div className="row">
        <input
          type="file"
          accept=".mp3,.mp4,.m4a,.wav,.ogg,.opus,.flac,.webm,.mkv,.mov,.aac,audio/*,video/*"
          disabled={!!busy}
          onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
        />
        {busy === "upload" && <span className="muted">Extracting audio…</span>}
      </div>
      {upload && (
        <div className="col" style={{ marginTop: 12 }}>
          <div className="small muted">Drag on the waveform to choose the part to clone ({upload.duration.toFixed(1)}s total).</div>
          <Waveform peaks={upload.peaks} duration={upload.duration} start={sel[0]} end={sel[1]}
            playhead={playhead} onChange={(s, e) => setSel([s, e])} />
          <div className="row small">
            <span>Selection: {sel[0].toFixed(1)}s – {sel[1].toFixed(1)}s ({selLen.toFixed(1)}s)</span>
            {selLen > 0 && (selLen < 8 || selLen > 40) && (
              <span className="warn-text">Aim for 10–30 s of clean speech.</span>
            )}
            <span className="grow" />
            <button className="btn sm" onClick={playSelection}>▶ Play selection</button>
            <button className="btn sm ghost" onClick={() => setSel([0, upload.duration])}>Select all</button>
          </div>
          <div className="grid3">
            <label className="field">Name<input value={name} onChange={(e) => setName(e.target.value)} /></label>
            <label className="field">Engine
              <select value={engine} onChange={(e) => setEngine(e.target.value)}>
                {engines.map((e) => (
                  <option key={e.id} value={e.id} disabled={!e.available}>{e.name}{e.available ? "" : " (not installed)"}</option>
                ))}
              </select>
            </label>
            <label className="field">Language
              <select value={language} onChange={(e) => setLanguage(e.target.value)}>
                {Object.entries(langs).map(([k, n]) => <option key={k} value={k}>{n}</option>)}
              </select>
            </label>
          </div>
          <div className="row end">
            <button className="btn ghost" onClick={() => setUpload(null)}>Cancel</button>
            <button className="btn primary" disabled={busy === "create" || selLen < 3} onClick={create}>
              {busy === "create" ? "Saving…" : "Save voice"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function VoiceCard({ voice, active, engines, editing, onEdit, onChange }: {
  voice: Voice; active: boolean; engines: EngineInfo[]; editing: boolean; onEdit: () => void; onChange: () => void;
}) {
  const [previewText, setPreviewText] = useState("");
  const [busy, setBusy] = useState(false);
  const [showTune, setShowTune] = useState(false);
  const toast = useToast();

  async function preview(throughMic: boolean) {
    setBusy(true);
    try {
      const blob = await api.previewVoice(voice.id, previewText, throughMic);
      if (!throughMic) await player.playBlob(blob);
    } catch (e) {
      toast(errMsg(e), "error");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className={`card ${active ? "active-voice" : ""}`}>
      <div className="voice-card">
        <div className="col" style={{ gap: 4 }}>
          <div className="row">
            <strong>{voice.name}</strong>
            {active && <span className="badge accent">active</span>}
            <span className="badge">{LANGUAGE_NAMES[voice.language] ?? voice.language}</span>
            <span className="badge">{engines.find((e) => e.id === voice.engine)?.name ?? voice.engine}</span>
            <span className="small muted">{voice.duration.toFixed(1)}s reference</span>
            {voice.vc_finetune && <span className="badge accent" title="Voice changer fine-tuned">fine-tuned</span>}
          </div>
        </div>
        <div className="row">
          {!active && <button className="btn sm" onClick={async () => { await api.setActiveVoice(voice.id); onChange(); }}>Use</button>}
          <button className="btn sm" onClick={() => void new Audio(`/api/voices/${voice.id}/reference`).play()}>Reference</button>
          <button className="btn sm" onClick={onEdit}>{editing ? "Close" : "Edit"}</button>
          <button className="btn sm" onClick={() => setShowTune(!showTune)}
            title="Train the real-time voice changer on more audio of this voice">
            {showTune ? "Close fine-tune" : "Fine-tune"}
          </button>
          <a className="btn sm" href={`/api/voices/${voice.id}/export`} download>Export</a>
          <button className="btn sm danger" onClick={async () => {
            if (!confirm(`Delete voice "${voice.name}"?`)) return;
            await api.deleteVoice(voice.id);
            onChange();
          }}>Delete</button>
        </div>
      </div>
      <div className="row" style={{ marginTop: 10 }}>
        <input className="grow" placeholder="Preview text (leave empty for a sample sentence)" value={previewText}
          onChange={(e) => setPreviewText(e.target.value)} />
        <button className="btn sm" disabled={busy} onClick={() => preview(false)}>{busy ? "Generating…" : "▶ Preview"}</button>
        <button className="btn sm" disabled={busy} onClick={() => preview(true)} title="Play through the virtual mic / monitor">🎚️ Through mic</button>
      </div>
      {editing && <VoiceEditor voice={voice} engines={engines} onSaved={onChange} />}
      {showTune && <FineTunePanel voice={voice} onChange={onChange} />}
    </div>
  );
}

function VoiceEditor({ voice, engines, onSaved }: { voice: Voice; engines: EngineInfo[]; onSaved: () => void }) {
  const [name, setName] = useState(voice.name);
  const [language, setLanguage] = useState(voice.language);
  const [engine, setEngine] = useState(voice.engine);
  const [settings, setSettings] = useState<Record<string, Record<string, any>>>(voice.engine_settings || {});
  const toast = useToast();
  const eng = engines.find((e) => e.id === engine);
  const values = settings[engine] || {};

  const setValue = (key: string, v: any) => setSettings((s) => ({ ...s, [engine]: { ...(s[engine] || {}), [key]: v } }));

  return (
    <div className="col" style={{ marginTop: 12 }}>
      <hr style={{ margin: "4px 0" }} />
      <div className="grid3">
        <label className="field">Name<input value={name} onChange={(e) => setName(e.target.value)} /></label>
        <label className="field">Engine
          <select value={engine} onChange={(e) => setEngine(e.target.value)}>
            {engines.map((e) => <option key={e.id} value={e.id} disabled={!e.available}>{e.name}</option>)}
          </select>
        </label>
        <label className="field">Language
          <select value={language} onChange={(e) => setLanguage(e.target.value)}>
            {Object.entries(eng?.languages ?? { [language]: language }).map(([k, n]) => <option key={k} value={k}>{n}</option>)}
          </select>
        </label>
      </div>
      {eng?.settings && (
        <div className="grid2">
          {eng.settings.map((s) => (
            <SettingInput key={s.key} spec={s} value={values[s.key] ?? s.default} onChange={(v) => setValue(s.key, v)} />
          ))}
        </div>
      )}
      <div className="row end">
        <button className="btn ghost" onClick={() => setSettings((s) => ({ ...s, [engine]: {} }))}>Reset engine settings</button>
        <button className="btn primary" onClick={async () => {
          try {
            await api.updateVoice(voice.id, { name, language, engine, engine_settings: settings });
            toast("Voice updated");
            onSaved();
          } catch (e) {
            toast(errMsg(e), "error");
          }
        }}>Save</button>
      </div>
    </div>
  );
}

export function SettingInput({ spec, value, onChange }: { spec: SettingSpec; value: any; onChange: (v: any) => void }) {
  if (spec.type === "choice") {
    return (
      <label className="field" title={spec.help}>{spec.label}
        <select value={value} onChange={(e) => onChange(e.target.value)}>
          {spec.choices?.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
        {spec.help && <span className="small">{spec.help}</span>}
      </label>
    );
  }
  if (spec.type === "bool") {
    return <label className="check"><input type="checkbox" checked={!!value} onChange={(e) => onChange(e.target.checked)} />{spec.label}</label>;
  }
  return (
    <label className="field" title={spec.help}>
      <span className="row between"><span>{spec.label}</span><span>{Number(value).toFixed(spec.type === "int" ? 0 : 2)}</span></span>
      <input type="range" min={spec.min} max={spec.max} step={spec.step ?? (spec.type === "int" ? 1 : 0.01)} value={value}
        onChange={(e) => onChange(spec.type === "int" ? parseInt(e.target.value) : parseFloat(e.target.value))} />
      {spec.help && <span className="small">{spec.help}</span>}
    </label>
  );
}

function ImportButton({ onImported }: { onImported: () => void }) {
  const ref = useRef<HTMLInputElement>(null);
  const toast = useToast();
  return (
    <>
      <input ref={ref} type="file" accept=".zip" style={{ display: "none" }} onChange={async (e) => {
        const f = e.target.files?.[0];
        if (!f) return;
        try {
          const v = await api.importVoice(f);
          toast(`Imported "${v.name}"`);
          onImported();
        } catch (err) {
          toast(errMsg(err), "error");
        }
        e.target.value = "";
      }} />
      <button className="btn sm" onClick={() => ref.current?.click()}>Import voice (.zip)</button>
    </>
  );
}

function EnginesPanel({ engines, onChange }: { engines: EngineInfo[]; onChange: () => void }) {
  const [busy, setBusy] = useState<string | null>(null);
  const toast = useToast();
  return (
    <>
      {engines.map((e) => (
        <div key={e.id} className="card tight">
          <div className="row">
            <strong>{e.name}</strong>
            {e.available ? (
              <span className={`pill ${e.loaded ? "good" : ""}`}>{e.loaded ? `loaded on ${e.device}${e.variant ? ` (${e.variant})` : ""}` : "not loaded"}</span>
            ) : (
              <span className="pill bad">not installed</span>
            )}
            {e.status.startsWith("error") && <span className="small error-text">{e.status}</span>}
            <span className="grow" />
            {e.available && !e.loaded && (
              <button className="btn sm" disabled={busy === e.id} onClick={async () => {
                setBusy(e.id);
                try {
                  await api.loadEngine(e.id);
                  toast(`${e.name} loaded`);
                } catch (err) {
                  toast(errMsg(err), "error");
                } finally {
                  setBusy(null);
                  onChange();
                }
              }}>{busy === e.id ? "Loading… (first time downloads the model)" : "Load now"}</button>
            )}
            {e.loaded && <button className="btn sm" onClick={async () => { await api.unloadEngine(e.id); onChange(); }}>Unload</button>}
          </div>
          <div className="small muted" style={{ marginTop: 6 }}>{e.available ? e.license : e.reason}</div>
        </div>
      ))}
    </>
  );
}
