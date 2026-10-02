import { useEffect, useMemo, useState } from "react";
import { api, LANGUAGE_NAMES, type LengthPreset, type Persona, type PersonaList, type Provider, type Voice } from "../api";
import { errMsg, useToast } from "../toast";

const EFFORTS = ["", "minimal", "low", "medium", "high"];

export default function PersonasPage({ voices }: { voices: Voice[] }) {
  const [data, setData] = useState<PersonaList | null>(null);
  const [providers, setProviders] = useState<Provider[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [draft, setDraft] = useState<Persona | null>(null);
  const [models, setModels] = useState<Record<string, string[]>>({});
  const toast = useToast();

  async function load(selectId?: string) {
    const [d, p] = await Promise.all([api.personas(), api.providers()]);
    setData(d);
    setProviders(p);
    const id = selectId ?? selected ?? d.active_id ?? d.personas[0]?.id ?? null;
    setSelected(id);
    setDraft(d.personas.find((x) => x.id === id) ?? null);
  }
  useEffect(() => {
    load().catch((e) => toast(errMsg(e), "error"));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const provider = providers.find((p) => p.id === draft?.provider);
  useEffect(() => {
    if (!provider?.configured || models[provider.id]) return;
    api.models(provider.id)
      .then((ms) => setModels((m) => ({ ...m, [provider.id]: ms.map((x) => x.id) })))
      .catch(() => setModels((m) => ({ ...m, [provider.id]: [] })));
  }, [provider, models]);

  const modelOptions = useMemo(() => {
    if (!provider) return [];
    return Array.from(new Set([...provider.suggested_models, ...(models[provider.id] ?? [])]));
  }, [provider, models]);

  if (!data || !draft) return <div className="page muted">Loading…</div>;
  const presets = data.length_presets;
  const dirty = JSON.stringify(draft) !== JSON.stringify(data.personas.find((p) => p.id === draft.id));
  const set = <K extends keyof Persona>(k: K, v: Persona[K]) => setDraft({ ...draft, [k]: v });

  async function save() {
    try {
      await api.updatePersona(draft!);
      toast("Persona saved");
      await load(draft!.id);
    } catch (e) {
      toast(errMsg(e), "error");
    }
  }

  return (
    <div className="page" style={{ display: "grid", gridTemplateColumns: "230px 1fr", gap: 20, alignItems: "start" }}>
      <div className="col">
        <h1>Personas</h1>
        {data.personas.map((p) => (
          <div key={p.id} className={`conv ${p.id === selected ? "active" : ""}`} onClick={() => {
            if (dirty && !confirm("Discard unsaved changes?")) return;
            setSelected(p.id);
            setDraft(p);
          }}>
            <span className="title">{p.name}</span>
            {p.id === data.active_id && <span className="badge accent">default</span>}
          </div>
        ))}
        <button className="btn" onClick={async () => {
          const p = await api.createPersona({ name: "New persona", provider: draft.provider, model: draft.model });
          await load(p.id);
        }}>+ New persona</button>
      </div>

      <div className="card page-narrow" style={{ margin: 0 }}>
        <div className="row between">
          <input style={{ fontSize: 16, fontWeight: 600, flex: 1 }} value={draft.name} onChange={(e) => set("name", e.target.value)} />
          <div className="row">
            {draft.id !== data.active_id && (
              <button className="btn sm" onClick={async () => { await api.activatePersona(draft.id); await load(draft.id); }}>Make default</button>
            )}
            <button className="btn sm" onClick={async () => { const p = await api.duplicatePersona(draft.id); await load(p.id); }}>Duplicate</button>
            <button className="btn sm danger" disabled={data.personas.length <= 1} onClick={async () => {
              if (!confirm(`Delete "${draft.name}"?`)) return;
              try {
                await api.deletePersona(draft.id);
                setSelected(null);
                await load();
              } catch (e) {
                toast(errMsg(e), "error");
              }
            }}>Delete</button>
          </div>
        </div>

        <h3 style={{ marginTop: 18 }}>System prompt</h3>
        <textarea rows={8} style={{ width: "100%" }} value={draft.system_prompt}
          placeholder="Who is the assistant? How should it talk? Anything it should always / never do?"
          onChange={(e) => set("system_prompt", e.target.value)} />

        <h3 style={{ marginTop: 18 }}>Answer length</h3>
        <div className="seg">
          {(["short", "normal", "long", "custom"] as LengthPreset[]).map((l) => (
            <button key={l} className={draft.length === l ? "on" : ""} onClick={() => set("length", l)}>
              {presets[l]?.label ?? "Custom"}
            </button>
          ))}
        </div>
        {draft.length !== "custom" ? (
          <p className="small muted">Adds: “{presets[draft.length]?.instruction}” · limit {presets[draft.length]?.max_tokens} tokens</p>
        ) : (
          <div className="grid2" style={{ marginTop: 10 }}>
            <label className="field">Length instruction
              <input value={draft.custom_length_instruction} placeholder="e.g. Answer in at most two sentences."
                onChange={(e) => set("custom_length_instruction", e.target.value)} />
            </label>
            <label className="field">Max tokens (hard cap)
              <input type="number" min={16} max={64000} value={draft.custom_max_tokens}
                onChange={(e) => set("custom_max_tokens", parseInt(e.target.value) || 800)} />
            </label>
          </div>
        )}
        <label className="check" style={{ marginTop: 8 }}>
          <input type="checkbox" checked={draft.voice_friendly} onChange={(e) => set("voice_friendly", e.target.checked)} />
          Speech-friendly output (no markdown, lists or emojis; written to be read aloud)
        </label>

        <h3 style={{ marginTop: 18 }}>Model</h3>
        <div className="grid2">
          <label className="field">Provider
            <select value={draft.provider} onChange={(e) => {
              const p = providers.find((x) => x.id === e.target.value);
              setDraft({ ...draft, provider: e.target.value, model: p?.default_model ?? draft.model });
            }}>
              {providers.map((p) => <option key={p.id} value={p.id}>{p.name}{p.configured ? "" : " (no API key)"}</option>)}
            </select>
          </label>
          <label className="field">Model
            <input list="model-options" value={draft.model} onChange={(e) => set("model", e.target.value)} />
            <datalist id="model-options">{modelOptions.map((m) => <option key={m} value={m} />)}</datalist>
          </label>
          <label className="field">
            <span className="row between">
              <span>Temperature</span>
              <label className="check small"><input type="checkbox" checked={draft.temperature === null}
                onChange={(e) => set("temperature", e.target.checked ? null : 0.7)} />model default</label>
            </span>
            <input type="range" min={0} max={2} step={0.05} disabled={draft.temperature === null}
              value={draft.temperature ?? 0.7} onChange={(e) => set("temperature", parseFloat(e.target.value))} />
            <span className="small">{draft.temperature === null ? "default" : draft.temperature.toFixed(2)} · ignored by models that don't support it</span>
          </label>
          <label className="field">Reasoning effort
            <select value={draft.reasoning_effort ?? ""} onChange={(e) => set("reasoning_effort", e.target.value || null)}>
              {EFFORTS.map((x) => <option key={x} value={x}>{x || "model default"}</option>)}
            </select>
            <span className="small">Lower = faster, cheaper replies. Good for live voice chat.</span>
          </label>
        </div>
        {provider && !provider.configured && <p className="warn-text small">Add a {provider.name} API key in Settings to use this persona.</p>}

        <h3 style={{ marginTop: 18 }}>Voice</h3>
        <div className="grid3">
          <label className="field">Voice
            <select value={draft.voice_id ?? ""} onChange={(e) => set("voice_id", e.target.value || null)}>
              <option value="">Active voice (sidebar)</option>
              {voices.map((v) => <option key={v.id} value={v.id}>{v.name}</option>)}
            </select>
          </label>
          <label className="field">Speech language
            <select value={draft.tts_language ?? ""} onChange={(e) => set("tts_language", e.target.value || null)}>
              <option value="">Voice's language</option>
              {Object.entries(LANGUAGE_NAMES).filter(([k]) => k !== "zh-cn").map(([k, n]) => <option key={k} value={k}>{n}</option>)}
            </select>
          </label>
          <label className="check" style={{ alignSelf: "end", paddingBottom: 8 }}>
            <input type="checkbox" checked={draft.speak_replies} onChange={(e) => set("speak_replies", e.target.checked)} />Speak replies
          </label>
        </div>

        <div className="row end" style={{ marginTop: 16 }}>
          {dirty && <span className="small warn-text">Unsaved changes</span>}
          <button className="btn ghost" disabled={!dirty} onClick={() => setDraft(data.personas.find((p) => p.id === draft.id) ?? draft)}>Revert</button>
          <button className="btn primary" disabled={!dirty} onClick={save}>Save persona</button>
        </div>
      </div>
    </div>
  );
}
