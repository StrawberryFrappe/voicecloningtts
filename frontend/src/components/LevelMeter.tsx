export default function LevelMeter({ value, label }: { value: number; label?: string }) {
  // value is a linear peak 0..1; display on a dB-ish scale so speech is visible.
  const db = value > 0 ? 20 * Math.log10(value) : -60;
  const pct = Math.max(0, Math.min(100, ((db + 60) / 60) * 100));
  return (
    <div className="row" style={{ gap: 8 }}>
      {label && <span className="small muted" style={{ width: 44 }}>{label}</span>}
      <div className="meter grow"><div style={{ width: `${pct}%` }} /></div>
    </div>
  );
}
