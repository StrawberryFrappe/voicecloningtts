import { useEffect, useState } from "react";
import { api, type AudioStatus, type DeviceInfo, type MixerSettings, type Settings } from "../api";
import { useEvents } from "../events";
import { errMsg, useToast } from "../toast";
import LevelMeter from "../components/LevelMeter";
import VoiceChangerCard from "../components/VoiceChangerCard";
import type { Voice } from "../api";

interface Devices {
  available: boolean; reason: string; inputs: DeviceInfo[]; outputs: DeviceInfo[]; virtual: DeviceInfo | null;
}

export default function AudioPage({ onChange, voices }: { onChange: () => void; voices: Voice[] }) {
  const [devices, setDevices] = useState<Devices | null>(null);
  const [status, setStatus] = useState<AudioStatus | null>(null);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [levels, setLevels] = useState({ mic: 0, tts: 0, out: 0 });
  const [busy, setBusy] = useState(false);
  const toast = useToast();

  async function load() {
    const [d, s, st] = await Promise.all([api.devices(), api.audioStatus(), api.settings()]);
    setDevices(d);
    setStatus(s);
    setSettings(st);
  }
  useEffect(() => {
    load().catch((e) => toast(errMsg(e), "error"));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEvents((e) => {
    if (e.type === "levels" && e.audio) setLevels(e.audio);
    if (e.type === "engine") api.audioStatus().then(setStatus);
  });

  if (!devices || !status || !settings) return <div className="page muted">Loading…</div>;
  const cfg = status.config;
  const mix = status.settings;
  const virtualFound = !!devices.virtual;

  async function setConfig(patch: Partial<AudioStatus["config"]>) {
    try {
      setStatus(await api.audioConfig(patch));
    } catch (e) {
      toast(errMsg(e), "error");
    }
  }
  async function setMix(patch: Partial<MixerSettings>) {
    setStatus(await api.mixer(patch));
  }
  async function toggle() {
    setBusy(true);
    try {
      setStatus(status!.running ? await api.audioStop() : await api.audioStart());
      onChange();
    } catch (e) {
      toast(errMsg(e), "error");
    } finally {
      setBusy(false);
    }
  }

  const outputs = devices.outputs;
  return (
    <div className="page page-narrow">
      <h1>Virtual microphone</h1>
      <p className="muted">
        The app mixes your real microphone with the cloned voice and plays the result into a virtual audio cable.
        In Discord, Zoom, OBS, games and so on, choose the cable's <b>recording</b> side as your microphone.
      </p>

      {!devices.available && <div className="card error-text">Audio devices unavailable: {devices.reason}</div>}

      <div className="card">
        <div className="row between">
          <h2 style={{ margin: 0 }}>Virtual cable</h2>
          {virtualFound ? <span className="pill good">found: {devices.virtual!.name}</span> : <span className="pill warn">not installed</span>}
        </div>
        {!virtualFound ? (
          <ol className="steps">
            <li>Download <b>VB-CABLE Virtual Audio Device</b> (free) from <a href="https://vb-audio.com/Cable/" target="_blank" rel="noreferrer">vb-audio.com/Cable</a>.</li>
            <li>Unzip it, right-click <code>VBCABLE_Setup_x64.exe</code> → <b>Run as administrator</b> → Install Driver.</li>
            <li>Reboot, then come back here and press <b>Refresh devices</b>.</li>
            <li>In Discord/Zoom select <b>CABLE Output (VB-Audio Virtual Cable)</b> as the input device.</li>
          </ol>
        ) : (
          <p className="small muted" style={{ marginBottom: 0 }}>
            Other apps should use <b>{devices.virtual!.name.replace(/Input/i, "Output")}</b> as their microphone.
            Turn off noise suppression in Discord ("Krisp") if the voice gets cut.
          </p>
        )}
      </div>

      <div className="card">
        <div className="row between">
          <h2 style={{ margin: 0 }}>Devices</h2>
          <button className="btn sm" onClick={() => load()}>Refresh devices</button>
        </div>
        <div className="grid3" style={{ marginTop: 12 }}>
          <DeviceSelect label="Your microphone" devices={devices.inputs} value={cfg.input_device}
            onChange={(v) => setConfig({ input_device: v })} noneLabel="— no mic (voice only) —" />
          <DeviceSelect label="Virtual cable (output)" devices={outputs} value={cfg.output_device}
            onChange={(v) => setConfig({ output_device: v })} highlight="virtual" />
          <DeviceSelect label="Monitor (your headphones)" devices={outputs} value={cfg.monitor_device}
            onChange={(v) => setConfig({ monitor_device: v })} noneLabel="— don't monitor —" />
        </div>
        {!cfg.output_device && virtualFound && (
          <button className="btn sm" style={{ marginTop: 10 }} onClick={() => setConfig({ output_device: devices.virtual!.name })}>
            Use {devices.virtual!.name}
          </button>
        )}
        <div className="row" style={{ marginTop: 14 }}>
          <button className={`btn ${status.running ? "danger" : "primary"}`} disabled={busy || !devices.available} onClick={toggle}>
            {status.running ? "■ Stop virtual mic" : "▶ Start virtual mic"}
          </button>
          <button className="btn" onClick={async () => {
            const r = await api.testTone();
            toast(r.sinks.length ? `Test tone → ${r.sinks.join(", ")}` : "No active output", r.sinks.length ? "info" : "warn");
          }}>Test tone</button>
          <label className="check small">
            <input type="checkbox" checked={settings.auto_start_audio}
              onChange={async (e) => setSettings(await api.saveSettings({ auto_start_audio: e.target.checked }))} />
            Start automatically with the app
          </label>
          {status.error && <span className="small error-text">{status.error}</span>}
        </div>
      </div>

      <VoiceChangerCard voices={voices} audioRunning={status.running} hasMic={!!cfg.input_device} />

      <div className="card">
        <h2>Mixer</h2>
        <div className="col" style={{ gap: 6, marginBottom: 14 }}>
          <LevelMeter label="Mic" value={levels.mic} />
          <LevelMeter label="Voice" value={levels.tts} />
          <LevelMeter label="Out" value={levels.out} />
        </div>
        <div className="grid2">
          <Slider label="Mic gain" unit="dB" min={-24} max={12} step={0.5} value={mix.mic_gain_db} onChange={(v) => setMix({ mic_gain_db: v })} />
          <Slider label="Voice gain" unit="dB" min={-24} max={12} step={0.5} value={mix.tts_gain_db} onChange={(v) => setMix({ tts_gain_db: v })} />
          <Slider label="Duck mic while voice speaks" unit="dB" min={-60} max={0} step={1} value={mix.duck_db}
            onChange={(v) => setMix({ duck_db: v })} hint={mix.duck_db <= -60 ? "mute" : mix.duck_db === 0 ? "off" : undefined} />
          <Slider label="Monitor volume" unit="dB" min={-30} max={6} step={0.5} value={mix.monitor_gain_db} onChange={(v) => setMix({ monitor_gain_db: v })} />
        </div>
        <div className="row" style={{ marginTop: 10 }}>
          <label className="check"><input type="checkbox" checked={mix.mic_enabled} onChange={(e) => setMix({ mic_enabled: e.target.checked })} />Pass my mic through</label>
          <label className="check"><input type="checkbox" checked={mix.monitor_tts} onChange={(e) => setMix({ monitor_tts: e.target.checked })} />Hear the voice in my headphones</label>
          <label className="check"><input type="checkbox" checked={mix.monitor_mic} onChange={(e) => setMix({ monitor_mic: e.target.checked })} />Hear my own mic (converted when the voice changer is on)</label>
        </div>
      </div>

      <div className="card">
        <h2>Record button defaults</h2>
        <div className="grid2">
          <label className="field">Default source
            <select value={settings.record_source} onChange={async (e) => setSettings(await api.saveSettings({ record_source: e.target.value as Settings["record_source"] }))}>
              <option value="mic">Microphone</option>
              <option value="loopback">Headphones (what you hear)</option>
            </select>
          </label>
          <DeviceSelect
            label={settings.record_source === "mic" ? "Microphone to record" : "Output to capture"}
            devices={settings.record_source === "mic" ? devices.inputs : outputs}
            value={settings.record_device || null}
            noneLabel="— system default —"
            onChange={async (v) => setSettings(await api.saveSettings({ record_device: v ?? "" }))}
          />
        </div>
      </div>
    </div>
  );
}

function DeviceSelect({ label, devices, value, onChange, noneLabel = "— none —", highlight }: {
  label: string; devices: DeviceInfo[]; value: string | null; onChange: (v: string | null) => void; noneLabel?: string; highlight?: "virtual";
}) {
  const missing = value && !devices.some((d) => d.name === value);
  return (
    <label className="field">{label}
      <select value={value ?? ""} onChange={(e) => onChange(e.target.value || null)}>
        <option value="">{noneLabel}</option>
        {missing && <option value={value!}>{value} (disconnected)</option>}
        {devices.map((d) => (
          <option key={d.id} value={d.name}>
            {highlight === "virtual" && d.is_virtual ? "★ " : ""}{d.name}{d.is_default_input || d.is_default_output ? " (default)" : ""}
          </option>
        ))}
      </select>
    </label>
  );
}

function Slider({ label, unit, min, max, step, value, onChange, hint }: {
  label: string; unit: string; min: number; max: number; step: number; value: number; onChange: (v: number) => void; hint?: string;
}) {
  const [v, setV] = useState(value);
  useEffect(() => setV(value), [value]);
  return (
    <label className="field">
      <span className="row between"><span>{label}</span><span>{hint ?? `${v > 0 ? "+" : ""}${v} ${unit}`}</span></span>
      <input type="range" min={min} max={max} step={step} value={v}
        onChange={(e) => setV(parseFloat(e.target.value))}
        onPointerUp={() => onChange(v)} onKeyUp={() => onChange(v)} />
    </label>
  );
}
