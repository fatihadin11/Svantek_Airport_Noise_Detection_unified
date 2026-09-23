import { useEffect, useMemo, useRef, useState } from "react";
import { startRecording, stopRecording } from "../api/recordings";

function formatElapsed(totalSeconds) {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

export default function RecordButton({ onRecordingFinished }) {
  const [isRecording, setIsRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState(null);
  const [isBusy, setIsBusy] = useState(false);
  const [folder, setFolder] = useState("Genel");
  const [tag, setTag] = useState("");
  const [color, setColor] = useState("#f2a65a");
  const [isEditorOpen, setIsEditorOpen] = useState(false);
  const intervalRef = useRef(null);

  const summary = useMemo(() => {
    const safeFolder = folder?.trim() || "Genel";
    const safeTag = tag?.trim() || "etiket yok";
    return `${safeFolder} · ${safeTag}`;
  }, [folder, tag]);

  useEffect(() => {
    return () => clearInterval(intervalRef.current);
  }, []);

  async function handleClick() {
    setError(null);
    setIsBusy(true);
    try {
      if (!isRecording) {
        // Edge tarafı aktif canlı izleme açıksa önce onu güvenli biçimde durdurur.
        await startRecording({ folder, tag, color });
        setIsRecording(true);
        setElapsed(0);
        setIsEditorOpen(false);
        intervalRef.current = setInterval(() => setElapsed((prev) => prev + 1), 1000);
      } else {
        clearInterval(intervalRef.current);
        const result = await stopRecording();
        setIsRecording(false);
        setElapsed(0);
        onRecordingFinished(result.recording);
      }
    } catch (err) {
      setError(err.message);
      setIsRecording(false);
      clearInterval(intervalRef.current);
    } finally {
      setIsBusy(false);
    }
  }

  function handleSaveEditor(event) {
    event.preventDefault();
    setFolder((prev) => prev?.trim() || "Genel");
    setTag((prev) => prev?.trim() || "");
    setIsEditorOpen(false);
  }

  return (
    <div className="record-control record-control-compact">
      <div className="record-meta-summary" title="Yeni SVAN kaydı bu bilgilerle listeye eklenir.">
        <span className="record-meta-label">Kayıt Bilgileri:</span>
        <span className="record-meta-text">{summary}</span>
        <span
          className="record-meta-color"
          style={{ backgroundColor: color }}
          aria-label="Kayıt rengi"
        />
        <button
          type="button"
          className="record-meta-edit"
          onClick={() => setIsEditorOpen((prev) => !prev)}
          disabled={isRecording || isBusy}
        >
          Düzenle
        </button>
      </div>

      <button
        className={`record-button ${isRecording ? "is-recording" : ""}`}
        onClick={handleClick}
        disabled={isBusy}
        aria-pressed={isRecording}
      >
        <span className="record-button-dot" aria-hidden="true" />
        {isBusy && isRecording ? "Aktarılıyor…" : isRecording ? "Kaydı Durdur" : "SVAN Kaydı Al"}
      </button>

      {isRecording && <span className="record-elapsed mono">{formatElapsed(elapsed)}</span>}
      {error && <span className="record-error">{error}</span>}

      {isEditorOpen && (
        <form className="record-meta-popover" onSubmit={handleSaveEditor}>
          <div className="record-meta-popover-title">Yeni SVAN kaydı için bilgiler</div>

          <label className="record-meta-field">
            <span>Klasör</span>
            <input
              value={folder}
              onChange={(e) => setFolder(e.target.value)}
              disabled={isRecording || isBusy}
              placeholder="Genel"
            />
          </label>

          <label className="record-meta-field">
            <span>Etiket</span>
            <input
              value={tag}
              onChange={(e) => setTag(e.target.value)}
              disabled={isRecording || isBusy}
              placeholder="örn. otomatik test"
            />
          </label>

          <label className="record-meta-field">
            <span>Renk</span>
            <input
              type="color"
              value={color}
              onChange={(e) => setColor(e.target.value)}
              disabled={isRecording || isBusy}
            />
          </label>

          <div className="record-meta-hint">
            SVAN kaydı tamamlandığında pop-up açılmaz; kayıt doğrudan bu bilgilerle eklenir.
          </div>

          <div className="record-meta-actions">
            <button type="button" onClick={() => setIsEditorOpen(false)}>Vazgeç</button>
            <button type="submit">Kaydet</button>
          </div>
        </form>
      )}
    </div>
  );
}
