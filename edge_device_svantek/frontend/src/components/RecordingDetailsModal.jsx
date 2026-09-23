import React, { useState } from "react";

export default function RecordingDetailsModal({ recording, folders, isNew, onSave, onCancel }) {
  const [title, setTitle] = useState(recording.title || "");
  const [folder, setFolder] = useState(recording.folder || "Genel");
  const [tag, setTag] = useState(recording.tag || "");
  const [color, setColor] = useState(recording.color || "#f2a65a");

  return (
    <div style={{
      position: "fixed", top: 0, left: 0, right: 0, bottom: 0,
      backgroundColor: "rgba(0,0,0,0.7)", zIndex: 9999,
      display: "flex", alignItems: "center", justifyContent: "center"
    }}>
      <div style={{
        background: "var(--bg-panel)", padding: "24px", borderRadius: "12px",
        width: "400px", maxWidth: "90%", border: "1px solid var(--line)",
        boxShadow: "0 8px 32px rgba(0,0,0,0.5)"
      }}>
        {/* Duruma göre başlık değişiyor */}
        <h3 style={{ margin: "0 0 20px 0", color: "var(--text-main)" }}>
          {isNew ? "Yeni Ses Eklendi!" : "Kaydı Düzenle"}
        </h3>
        
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
            <label style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>Başlık</label>
            <input 
              type="text" value={title} onChange={(e) => setTitle(e.target.value)} 
              style={{ background: "var(--bg-panel-raised)", color: "white", border: "1px solid var(--line)", padding: "10px", borderRadius: "6px" }}
            />
          </div>
          
          <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
            <label style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>Hedef Klasör</label>
            <select 
              value={folder} onChange={(e) => setFolder(e.target.value)}
              style={{ background: "var(--bg-panel-raised)", color: "white", border: "1px solid var(--line)", padding: "10px", borderRadius: "6px", cursor: "pointer" }}
            >
              {folders.map(f => (
                <option key={f.name} value={f.name}>{f.name}</option>
              ))}
            </select>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
            <label style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>Etiket (İsteğe Bağlı)</label>
            <input 
              type="text" placeholder="örn: Uçak Sesi, Anons..." value={tag} onChange={(e) => setTag(e.target.value)} 
              style={{ background: "var(--bg-panel-raised)", color: "white", border: "1px solid var(--line)", padding: "10px", borderRadius: "6px" }}
            />
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
            <label style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>Renk Belirle</label>
            <input 
              type="color" value={color} onChange={(e) => setColor(e.target.value)} 
              style={{ background: "transparent", border: "none", cursor: "pointer", height: "40px", width: "100%", padding: 0 }}
            />
          </div>
        </div>

        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "24px" }}>
          <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>*İstediğiniz zaman değiştirebilirsiniz</span>
          <div style={{ display: "flex", gap: "12px" }}>
            {/* Duruma göre iptal butonu metni değişiyor */}
            <button onClick={onCancel} style={{ background: "transparent", border: "1px solid var(--line)", color: "var(--text-main)", padding: "8px 16px", borderRadius: "6px", cursor: "pointer" }}>
              {isNew ? "Atla" : "İptal"}
            </button>
            <button onClick={() => onSave({ title, folder, tag, color })} style={{ background: "var(--accent-record)", border: "none", color: "white", padding: "8px 20px", borderRadius: "6px", cursor: "pointer", fontWeight: "bold" }}>
              Kaydet
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}