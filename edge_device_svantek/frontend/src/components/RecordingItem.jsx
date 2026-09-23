import { useEffect, useRef, useState } from "react";
import { updateRecordingMeta } from "../api/recordings";

export const COLORS = [
  { key: "red",    hex: "#E8483A", label: "Kırmızı" },
  { key: "orange", hex: "#F2A65A", label: "Turuncu" },
  { key: "green",  hex: "#4CAF7A", label: "Yeşil"   },
  { key: "blue",   hex: "#5B8DEF", label: "Mavi"    },
  { key: "purple", hex: "#9B6EF2", label: "Mor"     },
  { key: "gray",   hex: "#8B8D93", label: "Gri"     },
];

function formatDuration(seconds) {
  if (!seconds && seconds !== 0) return "—";
  return `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
}

function formatDate(isoString) {
  const date = new Date(isoString);
  const today = new Date();
  if (date.toDateString() === today.toDateString())
    return date.toLocaleTimeString("tr-TR", { hour: "2-digit", minute: "2-digit" });
  return date.toLocaleDateString("tr-TR", { day: "2-digit", month: "short" });
}

export default function RecordingItem({ recording, isSelected, onSelect, onDelete, onUpdate }) {
  const [isEditing, setIsEditing] = useState(false);
  const [project, setProject]     = useState(recording.project || "");
  const [tagInput, setTagInput]   = useState("");
  const [tags, setTags]           = useState(recording.tags || []);
  const [color, setColor]         = useState(recording.color_label || null);
  const [isSaving, setIsSaving]   = useState(false);
  const panelRef = useRef(null);

  // Kayıt değişince (başka kayıt seçilince) düzenleme alanlarını sıfırla
  useEffect(() => {
    setProject(recording.project || "");
    setTags(recording.tags || []);
    setColor(recording.color_label || null);
    setIsEditing(false);
  }, [recording.id]);

  // Panel dışına tıklanınca kaydet ve kapat
  useEffect(() => {
    if (!isEditing) return;
    function onOutsideClick(e) {
      if (panelRef.current && !panelRef.current.contains(e.target)) {
        handleSave();
      }
    }
    document.addEventListener("mousedown", onOutsideClick);
    return () => document.removeEventListener("mousedown", onOutsideClick);
  }, [isEditing, project, tags, color]);

  async function handleSave() {
    setIsEditing(false);
    setIsSaving(true);
    try {
      const updated = await updateRecordingMeta(recording.id, {
        project: project || null,
        tags,
        color_label: color || "",
      });
      onUpdate(updated);
    } catch (e) {
      console.error("Güncelleme hatası:", e);
    } finally {
      setIsSaving(false);
    }
  }

  function addTag() {
    const t = tagInput.trim().toLowerCase().replace(/\s+/g, "-");
    if (t && !tags.includes(t)) setTags([...tags, t]);
    setTagInput("");
  }

  const colorHex = color ? COLORS.find((c) => c.key === color)?.hex : null;

  return (
    <div className="rec-item-wrap">
      {/* Ana satır */}
      <div
        className={`recording-item ${isSelected ? "is-selected" : ""}`}
        onClick={() => onSelect(recording)}
      >
        <div className="recording-item-main">
          <span
            className="source-dot"
            style={colorHex ? { background: colorHex } : undefined}
            data-source={!colorHex ? recording.source : undefined}
          />
          <div className="recording-item-text">
            <div className="recording-item-title">{recording.title}</div>
            <div className="recording-item-sub mono">
              {formatDate(recording.created_at)} · {formatDuration(recording.duration_sec)}
              {recording.project && (
                <span className="item-project"> · {recording.project}</span>
              )}
            </div>
            {tags.length > 0 && (
              <div className="item-tags-row">
                {tags.map((t) => (
                  <span key={t} className="item-tag">#{t}</span>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="item-actions">
          <button
            className={`item-edit-btn ${isEditing ? "is-active" : ""}`}
            aria-label="Düzenle"
            onClick={(e) => { e.stopPropagation(); setIsEditing((v) => !v); }}
          >
            ···
          </button>
          <button
            className="delete-button"
            aria-label="Sil"
            onClick={(e) => { e.stopPropagation(); onDelete(recording); }}
          >
            ✕
          </button>
        </div>
      </div>

      {/* Düzenleme paneli */}
      {isEditing && (
        <div className="edit-panel" ref={panelRef} onClick={(e) => e.stopPropagation()}>
          {/* Renk */}
          <div className="edit-row">
            <span className="edit-label">Renk</span>
            <div className="color-swatches">
              <button
                className={`color-swatch swatch-none ${!color ? "swatch-active" : ""}`}
                onClick={() => setColor(null)}
                aria-label="Renk yok"
              />
              {COLORS.map((c) => (
                <button
                  key={c.key}
                  className={`color-swatch ${color === c.key ? "swatch-active" : ""}`}
                  style={{ background: c.hex }}
                  onClick={() => setColor(c.key)}
                  aria-label={c.label}
                  title={c.label}
                />
              ))}
            </div>
          </div>

          {/* Proje */}
          <div className="edit-row">
            <span className="edit-label">Proje</span>
            <input
              className="edit-input"
              type="text"
              value={project}
              onChange={(e) => setProject(e.target.value)}
              placeholder="Proje adı…"
            />
          </div>

          {/* Etiketler */}
          <div className="edit-row">
            <span className="edit-label">Etiketler</span>
            <input
              className="edit-input"
              type="text"
              value={tagInput}
              onChange={(e) => setTagInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === ",") { e.preventDefault(); addTag(); }
              }}
              placeholder="Etiket yazıp Enter…"
            />
            {tags.length > 0 && (
              <div className="tag-chips">
                {tags.map((t) => (
                  <span key={t} className="tag-chip">
                    #{t}
                    <button
                      onClick={() => setTags(tags.filter((x) => x !== t))}
                      aria-label={`${t} etiketini kaldır`}
                    >×</button>
                  </span>
                ))}
              </div>
            )}
          </div>

          <button className="edit-save-btn" onClick={handleSave} disabled={isSaving}>
            {isSaving ? "Kaydediliyor…" : "Kaydet"}
          </button>
        </div>
      )}
    </div>
  );
}