// Typed client for the VoiceCloningTTS backend.

export type Role = "user" | "assistant" | "tool";

export interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

export interface Message {
  id: string;
  conversation_id: string;
  seq: number;
  role: Role;
  content: string;
  tool_calls: ToolCall[];
  tool_call_id: string | null;
  name: string | null;
  meta: Record<string, any>;
  created_at: number;
}

export interface Conversation {
  id: string;
  title: string;
  persona_id: string | null;
  created_at: number;
  updated_at: number;
}

export type LengthPreset = "short" | "normal" | "long" | "custom";

export interface Persona {
  id: string;
  name: string;
  system_prompt: string;
  length: LengthPreset;
  custom_max_tokens: number;
  custom_length_instruction: string;
  voice_friendly: boolean;
  temperature: number | null;
  reasoning_effort: string | null;
  provider: string;
  model: string;
  voice_id: string | null;
  speak_replies: boolean;
  tts_language: string | null;
}

export interface PersonaList {
  active_id: string | null;
  personas: Persona[];
  length_presets: Record<string, { label: string; instruction: string; max_tokens: number }>;
}

export interface Provider {
  id: string;
  name: string;
  default_model: string;
  suggested_models: string[];
  configured: boolean;
  key_source: "env" | "stored" | null;
}

export interface Voice {
  id: string;
  name: string;
  language: string;
  engine: string;
  engine_settings: Record<string, Record<string, any>>;
  duration: number;
  source_filename: string | null;
  notes: string;
  created_at: number;
  vc_finetune: { steps: number; clips?: number; seconds?: number; trained_at?: number } | null;
  vc_use_finetune: boolean;
}

export interface GpuStatus {
  cuda: boolean;
  name: string;
  vram_gb: number;
  cc: string | null;
  low_vram: boolean;
  precision: "fp16" | "fp32";
  memory_mode: "auto" | "low" | "normal";
  precision_pref: "auto" | "fp16" | "fp32";
  holders: string[];
  exclusive: string | null;
}

export interface TrainingClip { name: string; seconds: number; removable: boolean }

export interface TrainStatus {
  state: "idle" | "starting" | "running" | "cancelling" | "done" | "error" | "cancelled";
  voice_id?: string;
  step?: number;
  max_steps?: number;
  batch_size?: number;
  loss?: number | null;
  eta_s?: number | null;
  message?: string;
  error?: string | null;
  defaults: { steps: number; batch_size: number; min_steps: number; max_steps: number };
}

export interface Preset { label: string; settings: VCSettings; latency_ms: number }

export interface SettingSpec {
  key: string;
  label: string;
  type: "float" | "int" | "choice" | "bool";
  default: any;
  min?: number;
  max?: number;
  step?: number;
  choices?: string[];
  help?: string;
}

export interface EngineInfo {
  id: string;
  name: string;
  available: boolean;
  reason: string;
  license: string;
  loaded: boolean;
  status: string;
  languages?: Record<string, string>;
  settings?: SettingSpec[];
  device?: string;
  variant?: string;
}

export interface DeviceInfo {
  id: number;
  name: string;
  hostapi: string;
  max_input_channels: number;
  max_output_channels: number;
  default_samplerate: number;
  is_default_input: boolean;
  is_default_output: boolean;
  is_virtual: boolean;
}

export interface MixerSettings {
  mic_enabled: boolean;
  mic_gain_db: number;
  tts_gain_db: number;
  duck_db: number;
  monitor_tts: boolean;
  monitor_mic: boolean;
  monitor_gain_db: number;
  duck_release_ms: number;
}

export interface AudioConfig {
  input_device: string | null;
  output_device: string | null;
  monitor_device: string | null;
  sample_rate: number;
  block_ms: number;
}

export interface AudioStatus {
  running: boolean;
  config: AudioConfig;
  settings: MixerSettings;
  levels: { mic: number; tts: number; out: number };
  speaking: boolean;
  error: string | null;
}

export interface AppStatus {
  version: string;
  platform: string;
  data_dir: string;
  secrets_backend: string;
  audio: AudioStatus & { available: boolean; reason: string };
  virtual_cable: { installed: boolean; device: string | null; capture_name?: string | null; reason: string };
  stt: { available: boolean; reason: string; device: string | null };
  loopback: { available: boolean; reason: string };
  vc: VCStatus;
  gpu: GpuStatus;
  training: TrainStatus;
  busy: boolean;
}

export interface VCSettings {
  diffusion_steps: number;
  inference_cfg_rate: number;
  max_prompt_length: number;
  block_time: number;
  crossfade_time: number;
  extra_time_ce: number;
  extra_time: number;
  extra_time_right: number;
  gate_db: number;
  hangover_blocks: number;
}

export interface VCStatus {
  available: boolean;
  reason: string;
  state: "off" | "loading" | "live" | "error";
  active: boolean;
  error: string | null;
  voice_id: string | null;
  saved_voice_id?: string | null;
  model_loaded: boolean;
  finetuned?: boolean;
  precision?: string;
  settings: VCSettings;
  device: string;
  latency_ms: number;
  infer_ms: number;
  block_ms: number;
  load: number;
  dropped_blocks: number;
  underruns: number;
}

export interface Settings {
  history_limit: number;
  browser_playback: "auto" | "on" | "off";
  tts_device: string;
  xtts_license_accepted: boolean;
  stt_model: string;
  stt_device: string;
  stt_language: string;
  auto_start_audio: boolean;
  record_source: "mic" | "loopback";
  record_device: string;
  gpu_memory_mode: "auto" | "low" | "normal";
  gpu_precision: "auto" | "fp16" | "fp32";
}

export interface Overrides {
  length?: LengthPreset | null;
  custom_max_tokens?: number | null;
  temperature?: number | null;
  provider?: string | null;
  model?: string | null;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method, headers: {} };
  if (body instanceof FormData) {
    init.body = body;
  } else if (body !== undefined) {
    init.body = JSON.stringify(body);
    (init.headers as Record<string, string>)["Content-Type"] = "application/json";
  }
  const res = await fetch(path, init);
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const data = await res.json();
      msg = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
    } catch {
      /* not json */
    }
    throw new ApiError(res.status, msg);
  }
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) return (await res.json()) as T;
  return (await res.blob()) as unknown as T;
}

const get = <T,>(p: string) => request<T>("GET", p);
const post = <T,>(p: string, b?: unknown) => request<T>("POST", p, b ?? {});
const put = <T,>(p: string, b?: unknown) => request<T>("PUT", p, b ?? {});
const patch = <T,>(p: string, b?: unknown) => request<T>("PATCH", p, b ?? {});
const del = <T,>(p: string) => request<T>("DELETE", p);

export const api = {
  status: () => get<AppStatus>("/api/status"),
  settings: () => get<Settings>("/api/settings"),
  saveSettings: (s: Partial<Settings>) => put<Settings>("/api/settings", s),

  providers: () => get<Provider[]>("/api/providers"),
  setKey: (id: string, api_key: string) => put(`/api/providers/${id}/key`, { api_key }),
  deleteKey: (id: string) => del(`/api/providers/${id}/key`),
  models: (id: string) => get<{ id: string; name: string | null }[]>(`/api/providers/${id}/models`),

  personas: () => get<PersonaList>("/api/personas"),
  createPersona: (p: Partial<Persona>) => post<Persona>("/api/personas", p),
  updatePersona: (p: Persona) => put<Persona>(`/api/personas/${p.id}`, p),
  deletePersona: (id: string) => del(`/api/personas/${id}`),
  duplicatePersona: (id: string) => post<Persona>(`/api/personas/${id}/duplicate`),
  activatePersona: (id: string) => post(`/api/personas/${id}/activate`),

  conversations: () => get<Conversation[]>("/api/conversations"),
  createConversation: (persona_id?: string | null) =>
    post<Conversation>("/api/conversations", { persona_id: persona_id ?? null }),
  updateConversation: (id: string, data: Partial<Conversation>) =>
    patch<Conversation>(`/api/conversations/${id}`, data),
  deleteConversation: (id: string) => del(`/api/conversations/${id}`),
  messages: (id: string) => get<Message[]>(`/api/conversations/${id}/messages`),
  send: (id: string, text: string, opts: { persona_id?: string | null; overrides?: Overrides; speak?: boolean | null }) =>
    post<{ turn_id: string }>(`/api/conversations/${id}/send`, { text, ...opts }),
  regenerate: (id: string, opts: { overrides?: Overrides; speak?: boolean | null } = {}) =>
    post<{ turn_id: string }>(`/api/conversations/${id}/regenerate`, opts),
  stop: () => post("/api/stop"),
  speak: (text: string, voice_id?: string | null, interrupt = false) =>
    post<{ turn_id: string }>("/api/speak", { text, voice_id: voice_id ?? null, interrupt }),

  voices: () => get<{ active_id: string | null; voices: Voice[] }>("/api/voices"),
  setActiveVoice: (voice_id: string | null) => put("/api/voices/active", { voice_id }),
  uploadAudio: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return post<{ upload_id: string; filename: string; duration: number; peaks: number[] }>("/api/voices/upload", fd);
  },
  createVoice: (body: {
    upload_id: string; name: string; language: string; engine: string;
    start: number | null; end: number | null; engine_settings?: Record<string, Record<string, any>>;
  }) => post<{ voice: Voice; warnings: string[] }>("/api/voices", body),
  updateVoice: (id: string, body: Partial<Voice>) => patch<Voice>(`/api/voices/${id}`, body),
  deleteVoice: (id: string) => del(`/api/voices/${id}`),
  prepareVoice: (id: string) => post(`/api/voices/${id}/prepare`),
  previewVoice: (id: string, text?: string, play = false) =>
    post<Blob>(`/api/voices/${id}/preview`, { text: text || null, play }),
  importVoice: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return post<Voice>("/api/voices/import", fd);
  },

  engines: () => get<EngineInfo[]>("/api/tts/engines"),
  loadEngine: (id: string, variant?: string) => post(`/api/tts/engines/${id}/load`, { variant }),
  unloadEngine: (id: string) => post(`/api/tts/engines/${id}/unload`),

  devices: () =>
    get<{
      available: boolean; reason: string; inputs: DeviceInfo[]; outputs: DeviceInfo[]; virtual: DeviceInfo | null;
      virtual_status?: AppStatus["virtual_cable"];
    }>("/api/audio/devices"),
  audioStatus: () => get<AudioStatus>("/api/audio/status"),
  audioConfig: (c: Partial<AudioConfig>) => put<AudioStatus>("/api/audio/config", c),
  mixer: (m: Partial<MixerSettings>) => put<AudioStatus>("/api/audio/mixer", m),
  audioStart: () => post<AudioStatus>("/api/audio/start"),
  audioStop: () => post<AudioStatus>("/api/audio/stop"),
  testTone: () => post<{ ok: boolean; sinks: string[] }>("/api/audio/test-tone"),

  recordStart: (source: string, device?: string | null) => post("/api/record/start", { source, device: device || null }),
  recordStop: (transcribe = true, language?: string | null) =>
    post<{ text?: string; duration: number; language?: string }>("/api/record/stop", { transcribe, language: language ?? null }),
  recordCancel: () => post("/api/record/cancel"),
  transcribeFile: (blob: Blob, filename = "clip.webm") => {
    const fd = new FormData();
    fd.append("file", blob, filename);
    return post<{ text: string; duration: number; language?: string }>("/api/stt/transcribe", fd);
  },
  sttModels: () => get<{ models: string[] }>("/api/stt/models"),
  discord: () => get<Record<string, any>>("/api/integrations/discord"),

  vcStatus: () => get<VCStatus>("/api/vc"),
  vcSettings: (settings: Partial<VCSettings>, voice_id?: string | null) =>
    put<VCStatus>("/api/vc/settings", { settings, voice_id: voice_id ?? null }),
  vcStart: (voice_id?: string | null) => post<VCStatus>("/api/vc/start", { voice_id: voice_id ?? null }),
  vcStop: () => post<VCStatus>("/api/vc/stop"),
  vcUnload: () => post<VCStatus>("/api/vc/unload"),
  vcPresets: () => get<{ presets: Record<string, Preset>; recommended: string }>("/api/vc/presets"),
  vcAutotune: (voice_id?: string | null) =>
    post<{ preset: string; ok: boolean; tried: { preset: string; infer_ms: number; block_ms: number; load: number }[]; status: VCStatus }>(
      "/api/vc/autotune", { voice_id: voice_id ?? null }),

  gpu: () => get<GpuStatus>("/api/gpu"),
  trainingClips: (vid: string) => get<{ clips: TrainingClip[]; seconds: number }>(`/api/voices/${vid}/training`),
  addTraining: (vid: string, upload_id: string) =>
    post<{ clips: TrainingClip[]; seconds: number }>(`/api/voices/${vid}/training`, { upload_id }),
  deleteTraining: (vid: string, name: string) =>
    del<{ clips: TrainingClip[]; seconds: number }>(`/api/voices/${vid}/training/${encodeURIComponent(name)}`),
  startFinetune: (vid: string, steps: number, batch_size?: number) =>
    post<TrainStatus>(`/api/voices/${vid}/finetune`, { steps, batch_size: batch_size ?? null }),
  deleteFinetune: (vid: string) => del<Voice>(`/api/voices/${vid}/finetune`),
  trainStatus: () => get<TrainStatus>("/api/vc/train"),
  cancelTrain: () => post<TrainStatus>("/api/vc/train/cancel"),
  compareFinetune: (vid: string, recording_id: string) =>
    post<{ base_wav_b64: string; finetuned_wav_b64: string }>(`/api/voices/${vid}/finetune/compare`, { recording_id }),
  recordStopRaw: () => post<{ recording_id: string; duration: number }>("/api/record/stop", { transcribe: false }),
};

export const LANGUAGE_NAMES: Record<string, string> = {
  en: "English", es: "Spanish", fr: "French", de: "German", it: "Italian", pt: "Portuguese",
  pl: "Polish", tr: "Turkish", ru: "Russian", nl: "Dutch", cs: "Czech", ar: "Arabic",
  zh: "Chinese", "zh-cn": "Chinese", ja: "Japanese", hu: "Hungarian", ko: "Korean", hi: "Hindi",
  da: "Danish", el: "Greek", fi: "Finnish", he: "Hebrew", ms: "Malay", no: "Norwegian",
  sv: "Swedish", sw: "Swahili",
};
