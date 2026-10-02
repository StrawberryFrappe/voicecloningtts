import { useEffect, useState } from "react";
import { api, type AppStatus, type Provider, type Settings } from "../api";
import { errMsg, useToast } from "../toast";

const KEY_LINKS: Record<string, string> = {
  openai: "https://platform.openai.com/api-keys",
  anthropic: "https://console.anthropic.com/settings/keys",
  gemini: "https://aistudio.google.com/apikey",
  openrouter: "https://openrouter.ai/keys",
};

export default function SettingsPage({ status, onChange }: { status: AppStatus | null; onChange: () => void }) {
  const [providers, setProviders] = useState<Provider[]>([]);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [sttModels, setSttModels] = useState<string[]>(["auto"]);
  const [discord, setDiscord] = useState<Record<string, any> | null>(null);
  const toast = useToast();

  async function load() {
    const [p, s, m, d] = await Promise.all([api.providers(), api.settings(), api.sttModels(), api.discord()]);
    setProviders(p);
    setSettings(s);
    setSttModels(m.models);
    setDiscord(d);
  }
  useEffect(() => {
    load().catch((e) => toast(errMsg(e), "error"));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (!settings) return <div className="page muted">Loading…</div>;
  const save = async (patch: Partial<Settings>) => {
    try {
      setSettings(await api.saveSettings(patch));
      onChange();
    } catch (e) {
      toast(errMsg(e), "error");
    }
  };

  return (
    <div className="page page-narrow">
      <h1>Settings</h1>

      <div className="card">
        <h2>API keys</h2>
        <p className="small muted">Keys are stored in {status?.secrets_backend === "file" ? "a local file in the app data folder" : "the Windows Credential Manager"} and never shown again.</p>
        {providers.map((p) => <KeyRow key={p.id} p={p} onChange={load} />)}
      </div>

      <div className="card">
        <h2>Voice engines</h2>
        <div className="grid2">
          <label className="field">Run TTS on
            <select value={settings.tts_device} onChange={(e) => save({ tts_device: e.target.value })}>
              <option value="auto">Auto (GPU if available)</option>
              <option value="cuda">NVIDIA GPU (CUDA)</option>
              <option value="cpu">CPU</option>
            </select>
          </label>
          <label className="field">Browser playback
            <select value={settings.browser_playback} onChange={(e) => save({ browser_playback: e.target.value as Settings["browser_playback"] })}>
              <option value="auto">Only when the virtual mic is off</option>
              <option value="on">Always (also play in this window)</option>
              <option value="off">Never</option>
            </select>
          </label>
        </div>
        <label className="check" style={{ marginTop: 12 }}>
          <input type="checkbox" checked={settings.xtts_license_accepted} onChange={(e) => save({ xtts_license_accepted: e.target.checked })} />
          <span>I accept the <a href="https://coqui.ai/cpml" target="_blank" rel="noreferrer">Coqui Public Model License</a> (non-commercial use only) to use XTTS-v2.</span>
        </label>
      </div>

      <div className="card">
        <h2>Speech-to-text (Record button)</h2>
        {!status?.stt.available && <p className="small warn-text">{status?.stt.reason}</p>}
        <div className="grid3">
          <label className="field">Whisper model
            <select value={settings.stt_model} onChange={(e) => save({ stt_model: e.target.value })}>
              {sttModels.map((m) => <option key={m} value={m}>{m === "auto" ? "Auto (large-v3-turbo on GPU)" : m}</option>)}
            </select>
          </label>
          <label className="field">Run on
            <select value={settings.stt_device} onChange={(e) => save({ stt_device: e.target.value })}>
              <option value="auto">Auto</option>
              <option value="cuda">GPU</option>
              <option value="cpu">CPU</option>
            </select>
          </label>
          <label className="field">Language
            <input placeholder="auto-detect (e.g. en, es)" value={settings.stt_language}
              onChange={(e) => setSettings({ ...settings, stt_language: e.target.value })}
              onBlur={(e) => save({ stt_language: e.target.value.trim() })} />
          </label>
        </div>
        {!status?.loopback.available && <p className="small muted">Headphone capture: {status?.loopback.reason}</p>}
      </div>

      <div className="card">
        <h2>Chat</h2>
        <label className="field" style={{ maxWidth: 260 }}>Messages of history sent to the model
          <input type="number" min={2} max={500} value={settings.history_limit}
            onChange={(e) => setSettings({ ...settings, history_limit: parseInt(e.target.value) || 40 })}
            onBlur={(e) => save({ history_limit: parseInt(e.target.value) || 40 })} />
        </label>
      </div>

      <div className="card">
        <h2>Discord bot <span className="badge">coming later</span></h2>
        <p className="small muted">
          The app is prepared to speak through its own Discord bot instead of your microphone (see
          <code>backend/vctts/integrations/discord/README.md</code>). Not active yet.
        </p>
        {discord && (
          <div className="kv">
            <div>discord.py installed</div><div>{discord.installed ? "yes" : "no"}</div>
            <div>Bot token</div><div>{discord.token_configured ? "configured" : "not set"}</div>
          </div>
        )}
      </div>

      {status && (
        <div className="card">
          <h2>About</h2>
          <div className="kv">
            <div>Version</div><div>{status.version}</div>
            <div>Data folder</div><div><code>{status.data_dir}</code></div>
            <div>Key storage</div><div>{status.secrets_backend}</div>
          </div>
        </div>
      )}
    </div>
  );
}

function KeyRow({ p, onChange }: { p: Provider; onChange: () => void }) {
  const [value, setValue] = useState("");
  const [testing, setTesting] = useState(false);
  const toast = useToast();
  return (
    <div className="row" style={{ marginBottom: 10 }}>
      <div style={{ width: 170 }}>
        <div>{p.name}</div>
        <a className="small" href={KEY_LINKS[p.id]} target="_blank" rel="noreferrer">get a key</a>
      </div>
      <input className="grow" type="password" autoComplete="off"
        placeholder={p.configured ? (p.key_source === "env" ? "set via environment variable" : "•••••••• saved") : "paste API key"}
        value={value} onChange={(e) => setValue(e.target.value)} />
      <button className="btn sm primary" disabled={!value.trim()} onClick={async () => {
        try {
          await api.setKey(p.id, value.trim());
          setValue("");
          toast(`${p.name} key saved`);
          onChange();
        } catch (e) {
          toast(errMsg(e), "error");
        }
      }}>Save</button>
      <button className="btn sm" disabled={!p.configured || testing} onClick={async () => {
        setTesting(true);
        try {
          const ms = await api.models(p.id);
          toast(`${p.name}: key works (${ms.length} models)`);
        } catch (e) {
          toast(errMsg(e), "error");
        } finally {
          setTesting(false);
        }
      }}>{testing ? "Testing…" : "Test"}</button>
      {p.configured && p.key_source === "stored" && (
        <button className="btn sm danger" onClick={async () => { await api.deleteKey(p.id); onChange(); }}>Remove</button>
      )}
    </div>
  );
}
