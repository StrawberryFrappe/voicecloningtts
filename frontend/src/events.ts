// Live event stream from the backend (WebSocket with auto-reconnect) and
// browser-side playback of preview audio.

import { useEffect, useRef, useState } from "react";

export type AppEvent = { type: string; [k: string]: any };
type Listener = (e: AppEvent) => void;

class EventStream {
  private ws: WebSocket | null = null;
  private listeners = new Set<Listener>();
  private retry = 0;
  connected = false;
  private stateListeners = new Set<(c: boolean) => void>();

  start() {
    if (this.ws) return;
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    this.ws = ws;
    ws.onopen = () => {
      this.retry = 0;
      this.setConnected(true);
    };
    ws.onmessage = (m) => {
      let ev: AppEvent;
      try {
        ev = JSON.parse(m.data);
      } catch {
        return;
      }
      this.listeners.forEach((l) => {
        try {
          l(ev);
        } catch (err) {
          console.error(err);
        }
      });
    };
    ws.onclose = () => {
      this.ws = null;
      this.setConnected(false);
      const delay = Math.min(5000, 300 * 2 ** this.retry++);
      window.setTimeout(() => this.start(), delay);
    };
    ws.onerror = () => ws.close();
  }

  private setConnected(c: boolean) {
    this.connected = c;
    this.stateListeners.forEach((l) => l(c));
  }

  onState(l: (c: boolean) => void) {
    this.stateListeners.add(l);
    return () => this.stateListeners.delete(l);
  }

  subscribe(l: Listener) {
    this.listeners.add(l);
    return () => {
      this.listeners.delete(l);
    };
  }

  send(msg: unknown) {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(msg));
  }
}

export const events = new EventStream();

export function useEvents(handler: Listener) {
  const ref = useRef(handler);
  ref.current = handler;
  useEffect(() => events.subscribe((e) => ref.current(e)), []);
}

export function useConnected() {
  const [c, setC] = useState(events.connected);
  useEffect(() => {
    const off = events.onState(setC);
    setC(events.connected); // the socket may have opened before we subscribed
    return () => {
      off();
    };
  }, []);
  return c;
}

// ---------------------------------------------------------------------------
// Browser playback of TTS audio (used when the virtual mic isn't running, or
// when "browser playback" is forced on in Settings).

class BrowserPlayer {
  private ctx: AudioContext | null = null;
  private nextTime = 0;
  private sources = new Set<AudioBufferSourceNode>();
  volume = 1;

  private context() {
    if (!this.ctx) this.ctx = new AudioContext();
    if (this.ctx.state === "suspended") void this.ctx.resume();
    return this.ctx;
  }

  async enqueueWavBase64(b64: string) {
    const ctx = this.context();
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const buf = await ctx.decodeAudioData(bytes.buffer.slice(0));
    this.enqueueBuffer(buf);
  }

  async playBlob(blob: Blob) {
    const ctx = this.context();
    const buf = await ctx.decodeAudioData(await blob.arrayBuffer());
    this.enqueueBuffer(buf);
  }

  private enqueueBuffer(buf: AudioBuffer) {
    const ctx = this.context();
    const src = ctx.createBufferSource();
    const gain = ctx.createGain();
    gain.gain.value = this.volume;
    src.buffer = buf;
    src.connect(gain).connect(ctx.destination);
    const start = Math.max(ctx.currentTime + 0.02, this.nextTime);
    src.start(start);
    this.nextTime = start + buf.duration;
    this.sources.add(src);
    src.onended = () => this.sources.delete(src);
  }

  stop() {
    this.sources.forEach((s) => {
      try {
        s.stop();
      } catch {
        /* already stopped */
      }
    });
    this.sources.clear();
    this.nextTime = 0;
  }

  get playing() {
    return this.sources.size > 0;
  }
}

export const player = new BrowserPlayer();

export function installPlayer() {
  return events.subscribe((e) => {
    if (e.type === "audio" && e.wav_b64) void player.enqueueWavBase64(e.wav_b64);
    else if (e.type === "audio_stop" || e.type === "stopped") player.stop();
  });
}
