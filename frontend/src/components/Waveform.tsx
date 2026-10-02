import { useEffect, useRef, useState } from "react";

interface Props {
  peaks: number[];
  duration: number;
  start: number;
  end: number;
  onChange: (start: number, end: number) => void;
  playhead?: number | null;
}

/** Waveform with a drag-to-select region (the part of the clip used for cloning). */
export default function Waveform({ peaks, duration, start, end, onChange, playhead }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const box = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<number | null>(null);

  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth * dpr;
    const h = c.clientHeight * dpr;
    c.width = w;
    c.height = h;
    const ctx = c.getContext("2d")!;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#6f7bd9";
    const n = peaks.length || 1;
    const bw = w / n;
    for (let i = 0; i < peaks.length; i++) {
      const ph = Math.max(1, peaks[i] * (h * 0.9));
      ctx.fillRect(i * bw, (h - ph) / 2, Math.max(1, bw - 0.5), ph);
    }
  }, [peaks]);

  const toTime = (clientX: number) => {
    const r = box.current!.getBoundingClientRect();
    return Math.max(0, Math.min(duration, ((clientX - r.left) / r.width) * duration));
  };

  return (
    <div
      ref={box}
      className="wave"
      onMouseDown={(e) => {
        const t = toTime(e.clientX);
        setDrag(t);
        onChange(t, t);
      }}
      onMouseMove={(e) => {
        if (drag === null) return;
        const t = toTime(e.clientX);
        onChange(Math.min(drag, t), Math.max(drag, t));
      }}
      onMouseUp={() => setDrag(null)}
      onMouseLeave={() => setDrag(null)}
    >
      <canvas ref={canvas} />
      {end > start && (
        <div className="sel" style={{ left: `${(start / duration) * 100}%`, width: `${((end - start) / duration) * 100}%` }} />
      )}
      {playhead != null && <div className="playhead" style={{ left: `${(playhead / duration) * 100}%` }} />}
    </div>
  );
}
