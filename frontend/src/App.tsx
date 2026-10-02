import { useCallback, useEffect, useState } from "react";
import { api, type AppStatus, type Voice } from "./api";
import { useConnected, useEvents } from "./events";
import { errMsg, ToastProvider, useToast } from "./toast";
import ChatPage from "./pages/ChatPage";
import VoicesPage from "./pages/VoicesPage";
import PersonasPage from "./pages/PersonasPage";
import AudioPage from "./pages/AudioPage";
import SettingsPage from "./pages/SettingsPage";

type Tab = "chat" | "voices" | "personas" | "audio" | "settings";
const TABS: { id: Tab; label: string; icon: string }[] = [
  { id: "chat", label: "Chat", icon: "💬" },
  { id: "voices", label: "Voices", icon: "🎙️" },
  { id: "personas", label: "Personas", icon: "🧠" },
  { id: "audio", label: "Virtual mic", icon: "🎚️" },
  { id: "settings", label: "Settings", icon: "⚙️" },
];

function Shell() {
  const [tab, setTab] = useState<Tab>(() => (localStorage.getItem("tab") as Tab) || "chat");
  const [status, setStatus] = useState<AppStatus | null>(null);
  const [voices, setVoices] = useState<{ active_id: string | null; voices: Voice[] }>({ active_id: null, voices: [] });
  const [speaking, setSpeaking] = useState(false);
  const [vcState, setVcState] = useState<string>("off");
  const connected = useConnected();
  const toast = useToast();

  const refresh = useCallback(async () => {
    try {
      const [s, v] = await Promise.all([api.status(), api.voices()]);
      setStatus(s);
      setVoices(v);
    } catch {
      /* backend restarting */
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh, connected]);
  useEffect(() => localStorage.setItem("tab", tab), [tab]);
  useEffect(() => {
    if (status?.vc) setVcState(status.vc.state);
  }, [status]);

  useEvents((e) => {
    if (e.type === "engine") void refresh();
    if (e.type === "vc") setVcState(e.state);
    if (e.type === "speaking") setSpeaking(!!e.active);
    if (e.type === "tts_start") setSpeaking(true);
    if (e.type === "tts_done" || e.type === "stopped") setSpeaking(false);
    if (e.type === "error" && !e.conversation_id) toast(e.message, "error");
  });

  const activeVoice = voices.voices.find((v) => v.id === voices.active_id);
  const micOn = status?.audio.running;

  return (
    <div className="app">
      <nav className="nav">
        <div className="brand"><span className="dot" />VoiceCloningTTS</div>
        {TABS.map((t) => (
          <button key={t.id} className={tab === t.id ? "active" : ""} onClick={() => setTab(t.id)}>
            <span>{t.icon}</span>{t.label}
          </button>
        ))}
        <div className="spacer" />
        <div className="status">
          <div className="row"><span className={`led ${connected ? "on" : ""}`} />{connected ? "Backend connected" : "Connecting…"}</div>
          <div className="row"><span className={`led ${micOn ? "on" : ""}`} />Virtual mic {micOn ? "on" : "off"}</div>
          <div className="row"><span className={`led ${speaking ? "speaking" : ""}`} />{speaking ? "Speaking…" : "Idle"}</div>
          {status?.vc?.available && (
            <div className="row">
              <span className={`led ${vcState === "live" ? "on" : ""}`} />
              Voice changer {vcState === "live" ? "on" : vcState === "loading" ? "loading…" : "off"}
              <button className="btn sm ghost" style={{ marginLeft: "auto", padding: "0 6px" }}
                title={vcState === "live" ? "Stop" : "Start (uses the voice chosen on the Virtual mic page)"}
                onClick={async () => {
                  try {
                    if (vcState === "live" || vcState === "loading") await api.vcStop();
                    else await api.vcStart();
                  } catch (err) {
                    toast(errMsg(err), "error");
                  }
                }}>{vcState === "live" || vcState === "loading" ? "■" : "▶"}</button>
            </div>
          )}
          <label className="field" style={{ marginTop: 6 }}>
            Active voice
            <select
              value={voices.active_id ?? ""}
              onChange={async (e) => {
                await api.setActiveVoice(e.target.value || null);
                void refresh();
              }}
            >
              <option value="">— none —</option>
              {voices.voices.map((v) => <option key={v.id} value={v.id}>{v.name}</option>)}
            </select>
          </label>
          {!activeVoice && voices.voices.length === 0 && (
            <span className="small warn-text">Create a voice in Voices.</span>
          )}
        </div>
      </nav>
      <main>
        {tab === "chat" && <ChatPage status={status} />}
        {tab === "voices" && <VoicesPage onChange={refresh} />}
        {tab === "personas" && <PersonasPage voices={voices.voices} />}
        {tab === "audio" && <AudioPage onChange={refresh} voices={voices.voices} />}
        {tab === "settings" && <SettingsPage status={status} onChange={refresh} />}
      </main>
    </div>
  );
}

export default function App() {
  return (
    <ToastProvider>
      <Shell />
    </ToastProvider>
  );
}
