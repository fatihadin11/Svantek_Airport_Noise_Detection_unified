import { useEffect, useRef, useState } from "react";
import { audioUrl, fetchWaveform, updateRecording } from "../api/recordings";
import CsvMultiGraph from "./CsvMultiGraph";
import RecordingAnalysis from "./RecordingAnalysis";

function SpeakerIcon({ level }) {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M4 9v6h4l5 5V4L8 9H4z" fill="currentColor" />
      {level >= 1 && <path d="M16 9a4 4 0 010 6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>}
      {level >= 2 && <path d="M19 6a8.5 8.5 0 010 12" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>}
      {level === 0 && <path d="M16 9l5 6M21 9l-5 6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>}
    </svg>
  );
}

function formatTime(totalSeconds) {
  if (!Number.isFinite(totalSeconds)) return "0:00";
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.floor(totalSeconds % 60);
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function encryptionLabel(recording) {
  if (!recording) return "";
  if (recording.encryption_status === "encrypted" && recording.plain_deleted) return "Şifreli";
  if (recording.encryption_status === "encrypted") return "Şifreli";
  if (recording.encryption_status === "encrypted_pending_verify") return "Doğrulama bekliyor";
  if (recording.encryption_status === "encrypting") return "Şifreleniyor";
  if (recording.encryption_status === "error") return "Şifreleme hatası";
  return "Şifresiz";
}

export default function WaveformPlayer({ recording, onUpdateSuccess }) {
  const canvasRef = useRef(null);
  const audioRef = useRef(null);
  const containerRef = useRef(null);

  const [points, setPoints] = useState(null);
  const [audioSrc, setAudioSrc] = useState(null);
  const [isAudioLoading, setIsAudioLoading] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [error, setError] = useState(null);
  const [volume, setVolume] = useState(() => {
    const saved = localStorage.getItem("playerVolume");
    return saved !== null ? Number(saved) : 1;
  });

  useEffect(() => {
    if (audioRef.current) audioRef.current.volume = volume;
    localStorage.setItem("playerVolume", String(volume));
  }, [volume, audioSrc]);

  useEffect(() => {
    if (!recording) return;
    let cancelled = false;
    setPoints(null);
    setProgress(0);
    setCurrentTime(0);
    setIsPlaying(false);
    setError(null);

    fetchWaveform(recording.id)
      .then((data) => { if (!cancelled) setPoints(data.points); })
      .catch((err) => { if (!cancelled) setError(err.message); });

    return () => { cancelled = true; };
  }, [recording]);

  useEffect(() => {
    if (!recording) return;
    let cancelled = false;
    let objectUrl = null;
    setAudioSrc(null);
    setIsAudioLoading(true);

    fetch(audioUrl(recording.id))
      .then((res) => { if (!res.ok) throw new Error("Ses dosyası indirilemedi"); return res.blob(); })
      .then((blob) => {
        if (cancelled) return;
        objectUrl = URL.createObjectURL(blob);
        setAudioSrc(objectUrl);
      })
      .catch((err) => { if (!cancelled) setError(err.message); })
      .finally(() => { if (!cancelled) setIsAudioLoading(false); });

    return () => { cancelled = true; if (objectUrl) URL.revokeObjectURL(objectUrl); };
  }, [recording]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !points) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    const gap = 1.5;
    const minBarSpan = 3;
    const barCount = Math.max(1, Math.min(points.length, Math.floor(width / minBarSpan)));
    const barWidth = Math.max(1, width / barCount - gap);
    const step = points.length / barCount;
    const displayPoints = [];

    for (let i = 0; i < barCount; i++) {
      const start = Math.floor(i * step);
      const end = Math.max(start + 1, Math.floor((i + 1) * step));
      let maxVal = 0;
      for (let j = start; j < end && j < points.length; j++) if (points[j] > maxVal) maxVal = points[j];
      displayPoints.push(maxVal);
    }

    const centerY = height / 2;
    const playedBars = Math.floor(progress * barCount);
    ctx.clearRect(0, 0, width, height);

    for (let i = 0; i < barCount; i++) {
      const amplitude = Math.max(0.04, displayPoints[i]);
      const barHeight = amplitude * height * 0.85;
      const x = i * (barWidth + gap);
      const y = centerY - barHeight / 2;
      ctx.fillStyle = i <= playedBars ? (recording.color || "#f2a65a") : "#3d414a";
      ctx.fillRect(x, y, barWidth, barHeight);
    }
  }, [points, progress, recording]);

  function handlePlayPause() {
    if (!audioRef.current) return;
    isPlaying ? audioRef.current.pause() : audioRef.current.play();
  }

  function handleTimeUpdate() {
    if (!audioRef.current || !audioRef.current.duration) return;
    setProgress(audioRef.current.currentTime / audioRef.current.duration);
    setCurrentTime(audioRef.current.currentTime);
  }

  function handleSeek(event) {
    if (!audioRef.current || !audioRef.current.duration || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
    audioRef.current.currentTime = ratio * audioRef.current.duration;
    setProgress(ratio);
    setCurrentTime(ratio * audioRef.current.duration);
  }

  function seekTo(seconds) {
    if (!audioRef.current) return;
    audioRef.current.currentTime = seconds;
    setCurrentTime(seconds);
    if (audioRef.current.duration) setProgress(seconds / audioRef.current.duration);
  }

  const handleFieldChange = async (fieldName, value) => {
    try {
      const updatedData = await updateRecording(recording.id, { [fieldName]: value });
      if (onUpdateSuccess) onUpdateSuccess(updatedData);
    } catch (err) {
      console.error("Güncelleme hatası:", err);
    }
  };

  if (!recording) return <div className="waveform-empty"><p>Dinlemek için soldaki listeden bir kayıt seçin.</p></div>;

  const sourceLabel = recording.source === "microphone" ? "Mikrofon" : recording.source === "svantek" ? "SVAN 971" : "Yüklendi";

  return (
    <div className="waveform-player" style={{ overflowY: "auto" }}>
      <div className="waveform-header">
        <div style={{ width: "100%" }}>
          <input
            type="text"
            className="waveform-title"
            defaultValue={recording.title}
            key={`title-${recording.id}`}
            onBlur={(e) => handleFieldChange("title", e.target.value)}
            style={{ background: "transparent", border: "none", color: "var(--text-main)", fontSize: "1.5rem", fontWeight: "bold", outline: "none", width: "100%", padding: 0 }}
          />
          <div className="waveform-meta mono" style={{ marginTop: "8px" }}>
            {new Date(recording.created_at).toLocaleString("tr-TR")} · {sourceLabel}
            {recording.folder && ` · ${recording.folder}`}
            {recording.has_csv && " · CSV var"}
            <span className={`encryption-badge waveform-encryption-badge ${recording.encryption_status === "encrypted" ? "is-encrypted" : "is-plain"}`}>
              {encryptionLabel(recording)}
            </span>
          </div>
        </div>
      </div>

      <div className="waveform-canvas-wrap" ref={containerRef} onClick={handleSeek}>
        {error && <div className="waveform-error">Yüklenemedi: {error}</div>}
        {!points && !error && <div className="waveform-loading">Waveform yükleniyor…</div>}
        <canvas ref={canvasRef} className="waveform-canvas" />
      </div>

      <CsvMultiGraph recording={recording} progress={progress} />

      <RecordingAnalysis recording={recording} onSeek={seekTo} />

      <div className="transport">
        <button className="play-button" onClick={handlePlayPause} disabled={!audioSrc || isAudioLoading}>{isPlaying ? "❚❚" : "▶"}</button>
        <span className="mono transport-time">{formatTime(currentTime)} / {formatTime(recording.duration_sec)}</span>
        <div className="volume-control">
          <button className="volume-icon-button" onClick={() => setVolume(v => v > 0 ? 0 : 1)}><SpeakerIcon level={volume === 0 ? 0 : volume < 0.5 ? 1 : 2} /></button>
          <input type="range" min="0" max="1" step="0.01" value={volume} onChange={(e) => setVolume(Number(e.target.value))} className="volume-slider" />
        </div>
      </div>

      {audioSrc && <audio ref={audioRef} src={audioSrc} onPlay={() => setIsPlaying(true)} onPause={() => setIsPlaying(false)} onEnded={() => setIsPlaying(false)} onTimeUpdate={handleTimeUpdate}/>} 
    </div>
  );
}
