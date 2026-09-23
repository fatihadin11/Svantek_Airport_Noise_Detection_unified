import { useEffect, useRef, useState } from "react";
import { fetchCsvViews, fetchCsvViewGraph } from "../api/recordings";

function formatAxisValue(value) {
  if (!Number.isFinite(value)) return "";
  const abs = Math.abs(value);
  if (abs >= 1000 || (abs > 0 && abs < 0.01)) return value.toExponential(1);
  if (abs >= 10) return value.toFixed(1);
  return value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
}

function seriesStroke(index, fallbackColor) {
  const palette = [fallbackColor, "#72d6c9", "#a7b7ff", "#ffcc66", "#ff7b7b", "#b68cff", "#6fd18c", "#f09cff"];
  return palette[index % palette.length];
}

function clearCanvas(canvas) {
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

export default function CsvMultiGraph({ recording, progress = 0 }) {
  const canvasRef = useRef(null);
  const [views, setViews] = useState([]);
  const [selected, setSelected] = useState(null);
  const [graph, setGraph] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    clearCanvas(canvasRef.current);
    setViews([]);
    setSelected(null);
    setGraph(null);
    setError(null);

    if (!recording?.has_csv) return;

    let cancelled = false;
    setIsLoading(true);
    fetchCsvViews(recording.id)
      .then((data) => {
        if (cancelled) return;
        const items = data.views || [];
        setViews(items);
        setSelected(items[0] || null);
      })
      .catch((err) => { if (!cancelled) setError(err.message); })
      .finally(() => { if (!cancelled) setIsLoading(false); });

    return () => { cancelled = true; };
  }, [recording]);

  useEffect(() => {
    clearCanvas(canvasRef.current);
    setGraph(null);
    setError(null);
    if (!recording?.has_csv || !selected) return;

    let cancelled = false;
    setIsLoading(true);
    fetchCsvViewGraph(recording.id, { view: selected.view, metric: selected.metric })
      .then((data) => { if (!cancelled) setGraph(data); })
      .catch((err) => { if (!cancelled) setError(err.message); })
      .finally(() => { if (!cancelled) setIsLoading(false); });

    return () => { cancelled = true; };
  }, [recording, selected]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const xValues = graph?.x_values || [];
    const series = graph?.series || [];
    if (!canvas) return;
    if (xValues.length === 0 || series.length === 0) {
      clearCanvas(canvas);
      return;
    }

    const styles = getComputedStyle(document.documentElement);
    const fallbackColor = recording?.color || styles.getPropertyValue("--accent-amber").trim() || "#f2a65a";
    const axisColor = styles.getPropertyValue("--line").trim() || "#3d414a";
    const gridColor = styles.getPropertyValue("--bg-panel-raised").trim() || "#242832";
    const textColor = styles.getPropertyValue("--text-muted").trim() || "#a7adb8";
    const cursorColor = styles.getPropertyValue("--text-faint").trim() || "#6f7480";

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    const padding = { top: 18, right: 18, bottom: 34, left: 62 };
    const plotWidth = Math.max(1, width - padding.left - padding.right);
    const plotHeight = Math.max(1, height - padding.top - padding.bottom);

    let xMin = Math.min(...xValues);
    let xMax = Math.max(...xValues);
    const allYValues = series.flatMap((item) => item.values).filter(Number.isFinite);
    let yMin = Math.min(...allYValues);
    let yMax = Math.max(...allYValues);
    if (xMax === xMin) xMax = xMin + 1;
    if (yMax === yMin) { yMin -= 1; yMax += 1; }
    const yPadding = (yMax - yMin) * 0.08;
    yMin -= yPadding;
    yMax += yPadding;

    const toCanvasX = (x) => padding.left + ((x - xMin) / (xMax - xMin)) * plotWidth;
    const toCanvasY = (y) => padding.top + (1 - ((y - yMin) / (yMax - yMin))) * plotHeight;

    ctx.clearRect(0, 0, width, height);
    ctx.font = "11px system-ui, sans-serif";

    for (let i = 0; i <= 4; i++) {
      const ratio = i / 4;
      const y = padding.top + ratio * plotHeight;
      const value = yMax - ratio * (yMax - yMin);
      ctx.strokeStyle = gridColor;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
      ctx.fillStyle = textColor;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(formatAxisValue(value), padding.left - 8, y);
    }

    ctx.strokeStyle = axisColor;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, height - padding.bottom);
    ctx.lineTo(width - padding.right, height - padding.bottom);
    ctx.stroke();

    ctx.fillStyle = textColor;
    ctx.textBaseline = "top";
    ctx.textAlign = "left";
    ctx.fillText(formatAxisValue(xMin), padding.left, height - padding.bottom + 8);
    ctx.textAlign = "right";
    ctx.fillText(formatAxisValue(xMax), width - padding.right, height - padding.bottom + 8);

    series.forEach((item, seriesIndex) => {
      ctx.strokeStyle = seriesStroke(seriesIndex, fallbackColor);
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      item.values.forEach((value, index) => {
        const x = toCanvasX(xValues[index]);
        const y = toCanvasY(value);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });

    if (Number.isFinite(progress) && progress > 0 && progress < 1) {
      const cursorX = padding.left + progress * plotWidth;
      ctx.strokeStyle = cursorColor;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(cursorX, padding.top);
      ctx.lineTo(cursorX, height - padding.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [graph, progress, recording]);

  const series = graph?.series || [];
  const pointCount = graph?.x_values?.length || 0;

  return (
    <div style={{ marginTop: "var(--space-4)", background: "var(--bg-panel)", border: "1px solid var(--line)", borderRadius: "var(--radius-md)", overflow: "hidden", flexShrink: 0 }}>
      <div style={{ padding: "var(--space-2) var(--space-3)", borderBottom: "1px solid var(--line)", display: "flex", justifyContent: "space-between", gap: "var(--space-3)", color: "var(--text-muted)", fontSize: "0.78rem" }}>
        <span>{selected?.label || "CSV grafikleri"}</span>
        {graph && <span className="mono">{graph.x_label} · {pointCount}{graph.rows > pointCount ? ` / ${graph.rows}` : ""} satır</span>}
      </div>

      {views.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: "6px", padding: "var(--space-2) var(--space-3)", borderBottom: "1px solid var(--line)" }}>
          {views.map((item) => (
            <button
              key={item.id}
              onClick={() => setSelected(item)}
              style={{
                border: "1px solid var(--line)",
                borderRadius: "999px",
                padding: "4px 10px",
                background: selected?.id === item.id ? "var(--bg-panel-raised)" : "transparent",
                color: "var(--text-main)",
                cursor: "pointer",
              }}
            >
              {item.label}
            </button>
          ))}
        </div>
      )}

      <div style={{ position: "relative", height: "240px", background: "var(--bg-deep)" }}>
        {isLoading && <div style={statusStyle}>CSV yükleniyor…</div>}
        {!isLoading && !recording?.has_csv && <div style={statusStyle}>Bu kayıt için CSV yok.</div>}
        {error && <div style={{ ...statusStyle, color: "var(--accent-record)" }}>CSV yüklenemedi: {error}</div>}
        {graph && series.length === 0 && <div style={statusStyle}>Bu görünüm için çizilebilir veri bulunamadı.</div>}
        <canvas ref={canvasRef} style={{ width: "100%", height: "100%", display: "block", opacity: series.length > 0 ? 1 : 0 }} />
      </div>

      {series.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: "8px", padding: "var(--space-2) var(--space-3)", color: "var(--text-muted)", fontSize: "0.75rem", maxHeight: "92px", overflowY: "auto" }}>
          {series.map((item, index) => (
            <span key={`${item.name}-${index}`} style={{ display: "inline-flex", alignItems: "center", gap: "5px" }}>
              <span style={{ width: "10px", height: "10px", borderRadius: "50%", background: seriesStroke(index, recording?.color || "#f2a65a"), display: "inline-block" }} />
              {item.name}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

const statusStyle = {
  position: "absolute",
  inset: 0,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  color: "var(--text-faint)",
  fontSize: "0.85rem",
  pointerEvents: "none",
  textAlign: "center",
  padding: "var(--space-4)",
};
