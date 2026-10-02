import { useCallback, useEffect, useRef, useState } from "react";
import { api, type AppStatus, type Conversation, type LengthPreset, type Message, type Persona } from "../api";
import { useEvents } from "../events";
import { errMsg, useToast } from "../toast";
import RecordButton from "../components/RecordButton";

const LENGTHS: { id: LengthPreset | ""; label: string }[] = [
  { id: "", label: "Persona" },
  { id: "short", label: "Short" },
  { id: "normal", label: "Normal" },
  { id: "long", label: "Long" },
];

export default function ChatPage({ status }: { status: AppStatus | null }) {
  const [convs, setConvs] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(() => localStorage.getItem("conv"));
  const [messages, setMessages] = useState<Message[]>([]);
  const [streaming, setStreaming] = useState<{ turn: string; text: string } | null>(null);
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [activePersona, setActivePersona] = useState<string | null>(null);
  const [length, setLength] = useState<LengthPreset | "">("");
  const [speak, setSpeak] = useState(true);
  const [text, setText] = useState("");
  const [speaking, setSpeaking] = useState(false);
  const [inlineError, setInlineError] = useState<string | null>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const toast = useToast();

  const active = convs.find((c) => c.id === activeId) || null;
  const personaId = active?.persona_id || activePersona;
  const persona = personas.find((p) => p.id === personaId);

  const loadConvs = useCallback(async () => {
    const list = await api.conversations();
    setConvs(list);
    return list;
  }, []);

  useEffect(() => {
    (async () => {
      const [list, ps] = await Promise.all([loadConvs(), api.personas()]);
      setPersonas(ps.personas);
      setActivePersona(ps.active_id);
      if (!list.find((c) => c.id === activeId)) setActiveId(list[0]?.id ?? null);
    })().catch((e) => toast(errMsg(e), "error"));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (activeId) localStorage.setItem("conv", activeId);
    setStreaming(null);
    setInlineError(null);
    if (!activeId) {
      setMessages([]);
      return;
    }
    api.messages(activeId).then(setMessages).catch(() => setMessages([]));
  }, [activeId]);

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight });
  }, [messages, streaming?.text]);

  useEvents((e) => {
    if (e.type === "tts_start") setSpeaking(true);
    if (e.type === "tts_done" || e.type === "stopped") setSpeaking(false);
    if (e.type === "conversation_updated") {
      setConvs((cs) => {
        const others = cs.filter((c) => c.id !== e.conversation.id);
        return [e.conversation, ...others].sort((a, b) => b.updated_at - a.updated_at);
      });
    }
    if (e.conversation_id !== activeId) return;
    switch (e.type) {
      case "message":
        setMessages((ms) => (ms.some((m) => m.id === e.message.id) ? ms : [...ms, e.message]));
        break;
      case "assistant_start":
        setInlineError(null);
        setStreaming({ turn: e.turn_id, text: "" });
        break;
      case "delta":
        setStreaming((s) => (s && s.turn === e.turn_id ? { ...s, text: s.text + e.text } : { turn: e.turn_id, text: e.text }));
        break;
      case "assistant_done":
        if (e.message) setMessages((ms) => (ms.some((m) => m.id === e.message.id) ? ms : [...ms, e.message]));
        setStreaming(null);
        break;
      case "history_changed":
        api.messages(activeId!).then(setMessages);
        break;
      case "error":
        setInlineError(e.message);
        break;
      case "warning":
        toast(e.message, "warn");
        break;
    }
  });

  async function newChat() {
    const c = await api.createConversation(activePersona);
    setConvs((cs) => [c, ...cs]);
    setActiveId(c.id);
  }

  async function send() {
    const t = text.trim();
    if (!t) return;
    let cid = activeId;
    try {
      if (!cid) {
        const c = await api.createConversation(activePersona);
        setConvs((cs) => [c, ...cs]);
        setActiveId(c.id);
        cid = c.id;
      }
      setText("");
      await api.send(cid, t, { persona_id: personaId, overrides: length ? { length } : undefined, speak });
    } catch (e) {
      setText(t);
      toast(errMsg(e), "error");
    }
  }

  async function speakOnly() {
    const t = text.trim();
    if (!t) return;
    try {
      await api.speak(t);
      setText("");
    } catch (e) {
      toast(errMsg(e), "error");
    }
  }

  const busy = !!streaming || speaking;
  const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant");

  return (
    <div className="chat">
      <aside className="convs">
        <div style={{ padding: 10 }}>
          <button className="btn primary" style={{ width: "100%", justifyContent: "center" }} onClick={newChat}>+ New chat</button>
        </div>
        <div className="list">
          {convs.map((c) => (
            <div key={c.id} className={`conv ${c.id === activeId ? "active" : ""}`} onClick={() => setActiveId(c.id)}>
              <span className="title" title={c.title}>{c.title}</span>
              <button
                className="btn ghost sm x"
                title="Delete chat"
                onClick={async (ev) => {
                  ev.stopPropagation();
                  if (!confirm(`Delete "${c.title}"?`)) return;
                  await api.deleteConversation(c.id);
                  const list = await loadConvs();
                  if (c.id === activeId) setActiveId(list[0]?.id ?? null);
                }}
              >✕</button>
            </div>
          ))}
          {convs.length === 0 && <div className="muted small" style={{ padding: 10 }}>No chats yet.</div>}
        </div>
      </aside>

      <section className="chat-main">
        <div className="chat-head">
          <label className="row small muted">
            Persona
            <select
              value={personaId ?? ""}
              onChange={async (e) => {
                const pid = e.target.value;
                if (active) {
                  const c = await api.updateConversation(active.id, { persona_id: pid });
                  setConvs((cs) => cs.map((x) => (x.id === c.id ? c : x)));
                } else {
                  await api.activatePersona(pid);
                  setActivePersona(pid);
                }
              }}
            >
              {personas.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
            </select>
          </label>
          <div className="seg" title="Answer length for the next messages">
            {LENGTHS.map((l) => (
              <button key={l.id} className={length === l.id ? "on" : ""} onClick={() => setLength(l.id)}>{l.label}</button>
            ))}
          </div>
          <label className="check small"><input type="checkbox" checked={speak} onChange={(e) => setSpeak(e.target.checked)} />Speak replies</label>
          <span className="grow" />
          {persona && <span className="badge">{persona.provider} · {persona.model}</span>}
          {status && !status.audio.running && (
            <span className="pill warn" title="Audio plays in this window. Start the virtual mic to send it to other apps.">preview audio</span>
          )}
          {busy && <button className="btn danger solid sm" onClick={() => api.stop()}>■ Stop</button>}
        </div>

        <div className="messages" ref={scroller}>
          {messages.length === 0 && !streaming && (
            <div className="empty">
              <h2>Start talking</h2>
              <p>Type a message or hit <b>Record</b> to dictate it. Replies are spoken with your active cloned voice
                {status?.audio.running ? " through the virtual microphone." : "."}</p>
            </div>
          )}
          {messages.map((m) => <MessageView key={m.id} m={m} />)}
          {streaming && <div className="msg assistant streaming">{streaming.text}</div>}
          {inlineError && <div className="msg tool error-text">⚠ {inlineError}</div>}
        </div>

        <div className="composer">
          <textarea
            placeholder="Message… (Enter to send, Shift+Enter for a new line)"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void send();
              }
            }}
          />
          <div className="row between">
            <RecordButton onText={(t) => setText((cur) => (cur ? cur + " " : "") + t)} />
            <div className="row">
              {lastAssistant && activeId && !streaming && (
                <button className="btn sm ghost" onClick={() => api.regenerate(activeId, { overrides: length ? { length } : undefined, speak }).catch((e) => toast(errMsg(e), "error"))}>
                  ↻ Regenerate
                </button>
              )}
              <button className="btn" disabled={!text.trim()} onClick={speakOnly} title="Say this text with the voice, without asking the LLM">
                🔊 Say it
              </button>
              <button className="btn primary" disabled={!text.trim()} onClick={send}>Send ➤</button>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

function MessageView({ m }: { m: Message }) {
  if (m.role === "tool") {
    return <div className="msg tool">🔧 {m.name}: {m.content.slice(0, 400)}</div>;
  }
  if (m.role === "assistant" && !m.content && m.tool_calls.length) {
    return <div className="msg tool">🔧 calling {m.tool_calls.map((t) => t.name).join(", ")}…</div>;
  }
  return (
    <div className={`msg ${m.role}`}>
      {m.content}
      {m.role === "assistant" && (m.meta.model || m.meta.interrupted) && (
        <div className="meta">
          {m.meta.model && <span>{m.meta.model}</span>}
          {m.meta.interrupted && <span className="warn-text">interrupted</span>}
        </div>
      )}
    </div>
  );
}
