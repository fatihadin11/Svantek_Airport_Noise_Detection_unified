import React, { useState } from "react";

function formatDuration(seconds) {
  if (!seconds && seconds !== 0) return "—";
  const minutes = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${minutes}:${String(secs).padStart(2, "0")}`;
}

function encryptionLabel(rec) {
  if (rec.encryption_status === "encrypted" && rec.plain_deleted) return "Şifreli";
  if (rec.encryption_status === "encrypted") return "Şifreli";
  if (rec.encryption_status === "encrypted_pending_verify") return "Doğrulama bekliyor";
  if (rec.encryption_status === "encrypting") return "Şifreleniyor";
  if (rec.encryption_status === "error") return "Şifreleme hatası";
  return "Şifresiz";
}

export default function RecordingList({
  recordings, folders, selectedId, onSelect, onDelete, onEdit, isLoading,
  onMoveRecording, onDeleteFolder
}) {
  const [dragOverFolder, setDragOverFolder] = useState(null);

  if (isLoading) return <div className="list-status">Yükleniyor…</div>;

  return (
    <div className="recording-list-container" style={{ padding: 0 }}>
      {folders.map((folderObj) => {
        const folderName = folderObj.name;
        const items = recordings.filter(r => (r.folder || 'Genel') === folderName);

        return (
          <div 
            key={folderName}
            onDragOver={(e) => { e.preventDefault(); setDragOverFolder(folderName); }}
            onDragLeave={() => setDragOverFolder(null)}
            onDrop={(e) => {
              e.preventDefault(); setDragOverFolder(null);
              const recId = e.dataTransfer.getData("recId");
              if(recId) onMoveRecording(parseInt(recId), folderName);
            }}
            style={{ 
              marginBottom: "8px", transition: "background 0.2s",
              backgroundColor: dragOverFolder === folderName ? "rgba(255,255,255,0.05)" : "transparent" 
            }}
          >
            {/* Klasör Başlığı */}
            <div style={{ 
              padding: '12px 16px 8px', background: 'var(--bg-panel)', 
              fontSize: '0.75rem', fontWeight: 'bold', color: 'var(--text-muted)',
              position: 'sticky', top: 0, zIndex: 2, borderBottom: '1px solid var(--line)',
              display: 'flex', justifyContent: 'space-between', alignItems: 'center'
            }}>
              <span style={{ textTransform: 'uppercase', letterSpacing: '0.05em' }}>📁 {folderName}</span>
              {folderName !== 'Genel' && (
                <button 
                  onClick={() => onDeleteFolder(folderName)} 
                  style={{ background: 'transparent', border: 'none', color: '#ff4a4a', cursor: 'pointer', fontSize: '0.7rem' }}
                  title="Klasörü Sil"
                >
                  Sil
                </button>
              )}
            </div>
            
            <ul className="recording-list" style={{ paddingBottom: '8px' }}>
              {items.length === 0 ? (
                <li style={{ 
                  margin: "8px 16px", padding: "16px", textAlign: "center", color: "var(--text-muted)", 
                  fontSize: "0.8rem", border: "1px dashed var(--line)", borderRadius: "6px" 
                }}>
                  Kayıtları buraya sürükleyin
                </li>
              ) : items.map((rec) => (
                <li
                  key={rec.id}
                  draggable
                  onDragStart={(e) => e.dataTransfer.setData("recId", rec.id)}
                  className={`recording-item ${rec.id === selectedId ? "is-selected" : ""}`}
                  onClick={() => onSelect(rec)}
                  style={{ cursor: "grab" }}
                >
                  <div className="recording-item-main">
                    <span className="source-dot" style={{ backgroundColor: rec.color || '#f2a65a' }} aria-hidden="true" />
                    <div className="recording-item-text">
                      <div className="recording-item-title">{rec.title}</div>
                      <div className="recording-item-sub mono" style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                        <span>{formatDuration(rec.duration_sec)}</span>
                        <span className={`encryption-badge ${rec.encryption_status === "encrypted" ? "is-encrypted" : "is-plain"}`}>
                          {encryptionLabel(rec)}
                        </span>
                        {rec.tag && (
                          <span style={{ 
                            background: 'var(--bg-panel-raised)', padding: '1px 6px', 
                            borderRadius: '4px', fontSize: '0.7rem', color: 'var(--text-main)', border: '1px solid var(--line)'
                          }}>🏷️ {rec.tag}</span>
                        )}
                      </div>
                    </div>
                  </div>
                  
                  {/* AKSİYON BUTONLARI (Düzenle ve Sil) */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <button
                      onClick={(e) => { e.stopPropagation(); onEdit(rec); }}
                      style={{ 
                        background: 'transparent', border: 'none', color: 'var(--text-muted)', 
                        cursor: 'pointer', fontSize: '1.2rem', padding: '0 6px', display: 'flex', alignItems: 'center' 
                      }}
                      title="Düzenle"
                    >
                      ⋮
                    </button>
                    <button className="delete-button" onClick={(e) => { e.stopPropagation(); onDelete(rec); }}>✕</button>
                  </div>

                </li>
              ))}
            </ul>
          </div>
        );
      })}
    </div>
  );
}