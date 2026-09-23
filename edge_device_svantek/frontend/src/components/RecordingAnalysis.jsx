import { useEffect, useState } from "react";
import { fetchRecordingAnalysis, startRecordingAnalysis } from "../api/recordings";

function formatTime(seconds) {
  const minute = Math.floor(seconds / 60);
  const second = Math.floor(seconds % 60);
  return `${minute}:${String(second).padStart(2, "0")}`;
}

export default function RecordingAnalysis({ recording, onSeek }) {
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState(null);

  async function load() {
    if (!recording) return;
    try {
      setAnalysis(await fetchRecordingAnalysis(recording.id));
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    setAnalysis(null);
    setError(null);
    load();
  }, [recording?.id]);

  useEffect(() => {
    if (analysis?.status !== "running") return undefined;
    const timer = setInterval(load, 2000);
    return () => clearInterval(timer);
  }, [analysis?.status, recording?.id]);

  async function start() {
    try {
      setAnalysis(await startRecordingAnalysis(recording.id));
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  }

  if (!recording) return null;
  const isRunning = analysis?.status === "running";
  const isComplete = analysis?.status === "completed";

  return (
    <section className="analysis-panel">
      <div className="analysis-heading">
        <div>
          <div className="analysis-kicker mono">KAYIT SONRASI AI ANALİZİ</div>
          <h3>Algılanan ses olayları</h3>
        </div>
        <button className="analysis-action" onClick={start} disabled={isRunning}>
          {isRunning ? "Analiz sürüyor…" : isComplete ? "Tekrar analiz et" : "AI ile analiz et"}
        </button>
      </div>

      {analysis?.model_name && <p className="analysis-meta mono">Model: {analysis.model_name}</p>}
      {analysis?.status === "not_started" && <p className="analysis-empty">Kayıt tamamlandı. AI, bu kaydın içindeki ses olaylarını bulmaya hazır.</p>}
      {isRunning && <p className="analysis-empty">Ses geçici olarak dönüştürülüyor ve olaylar çıkarılıyor…</p>}
      {(error || analysis?.error_message) && <p className="analysis-error">Analiz yapılamadı: {error || analysis.error_message}</p>}
      {isComplete && analysis.events.length === 0 && <p className="analysis-empty">Güven eşiğini geçen tanınmış bir ses olayı bulunamadı.</p>}

      {isComplete && analysis.events.length > 0 && (
        <div className="analysis-events">
          {analysis.events.map((event) => (
            <button key={event.id ?? `${event.start_sec}-${event.label}`} className="analysis-event" onClick={() => onSeek?.(event.start_sec)}>
              <span className="analysis-time mono">{formatTime(event.start_sec)}–{formatTime(event.end_sec)}</span>
              <span className="analysis-label">{event.label}</span>
              <span className="analysis-confidence">%{Math.round(event.confidence * 100)}</span>
            </button>
          ))}
        </div>
      )}
    </section>
  );
}
