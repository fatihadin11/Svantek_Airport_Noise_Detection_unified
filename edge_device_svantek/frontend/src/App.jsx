import { useEffect, useState } from "react";
import { fetchRecordings, deleteRecording, updateRecording, fetchFolders, createFolder, deleteFolder } from "./api/recordings";
import RecordingList from "./components/RecordingList";
import WaveformPlayer from "./components/WaveformPlayer";
import RecordButton from "./components/RecordButton";
import UploadButton from "./components/UploadButton";
import ConfirmDialog from "./components/ConfirmDialog";
import RecordingDetailsModal from "./components/RecordingDetailsModal";
import LiveMonitor from "./components/LiveMonitor";
import "./App.css";
import "./live.css";

export default function App() {
  const [recordings, setRecordings] = useState([]);
  const [folders, setFolders] = useState([]);
  const [selected, setSelected] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState(null);
  const [pendingDelete, setPendingDelete] = useState(null);
  const [activeTab, setActiveTab] = useState("recordings");

  const [editingRecording, setEditingRecording] = useState(null);
  const [isNewRecording, setIsNewRecording] = useState(false);

  // Tüm verileri (Klasörler ve Sesler) baştan çeken ana fonksiyon
  async function loadAllData() {
    setIsLoading(true);
    try {
      const [recsData, foldersData] = await Promise.all([
        fetchRecordings(),
        fetchFolders()
      ]);
      setRecordings(recsData);
      setFolders(foldersData);
      setLoadError(null);
    } catch (err) {
      setLoadError(err.message);
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    // 1. Uygulama ilk açıldığında listeyi yükle
    loadAllData();

    // 2. Arka planda her 5 saniyede bir yeni kayıt ve klasör var mı diye kontrol et.
    // Trigger kaydı yeni bir klasöre (ör. "Otomatik") düşerse klasör listesi de
    // yenilenmediğinde kayıt DB'de olsa bile panelde görünmüyordu.
    const intervalId = setInterval(async () => {
      try {
        const [recsData, foldersData] = await Promise.all([
          fetchRecordings(),
          fetchFolders()
        ]);

        setRecordings(recsData);
        setFolders(foldersData);
      } catch (err) {
        console.error("Arka plan yenileme hatası:", err);
      }
    }, 5000);

    // Bileşen kapatıldığında sayacı temizle
    return () => clearInterval(intervalId);
  }, []);

  function handleNewRecording(recording, options = {}) {
    setRecordings((prev) => {
      const exists = prev.some((rec) => rec.id === recording.id);
      if (exists) return prev.map((rec) => (rec.id === recording.id ? recording : rec));
      return [recording, ...prev];
    });

    // Kayıt yeni bir klasöre geldiyse, bir sonraki polling'i beklemeden
    // klasörü arayüze ekle.
    if (recording.folder) {
      setFolders((prev) => {
        const exists = prev.some((folder) => folder.name === recording.folder);
        if (exists) return prev;
        return [...prev, { name: recording.folder }];
      });
    }

    setSelected(recording);
    setActiveTab("recordings");

    // SVAN kayıtlarında kayıt öncesi "Kayıt Bilgileri" kullanıldığı için pop-up açmıyoruz.
    // Manuel upload tarafında kullanıcı isterse detay pop-up'ı açılabilir.
    if (options.openDetails) {
      setEditingRecording(recording);
      setIsNewRecording(Boolean(options.isNew));
    }
  }

  function handleEditRecording(recording) {
    setEditingRecording(recording);
    setIsNewRecording(false);
  }

  async function handleSaveDetails(details) {
    if (!editingRecording) return;
    try {
      const updated = await updateRecording(editingRecording.id, details);
      handleUpdateRecordingState(updated);
    } catch (err) {
      console.error("Detaylar kaydedilemedi", err);
    }
    setEditingRecording(null);
  }

  function handleUpdateRecordingState(updatedRec) {
    setRecordings((prev) =>
      prev.map((rec) => (rec.id === updatedRec.id ? updatedRec : rec))
    );
    if (selected && selected.id === updatedRec.id) {
      setSelected(updatedRec);
    }
  }

  async function handleMoveRecording(recId, targetFolder) {
    try {
      const updated = await updateRecording(recId, { folder: targetFolder });
      handleUpdateRecordingState(updated);
    } catch (err) {
      console.error("Taşıma başarısız oldu", err);
    }
  }

  async function handleCreateFolder() {
    const folderName = prompt("Yeni klasör adını girin:");
    if (!folderName || folderName.trim() === "") return;
    try {
      await createFolder(folderName.trim());
      const newFolders = await fetchFolders();
      setFolders(newFolders);
    } catch (err) {
      alert("Klasör oluşturulamadı.");
    }
  }

  async function handleDeleteFolder(folderName) {
    const confirmed = window.confirm(`"${folderName}" klasörünü silmek istediğinize emin misiniz?\nİçindeki sesler "Genel" klasörüne taşınacaktır.`);
    if (!confirmed) return;
    try {
      await deleteFolder(folderName);
      loadAllData();
    } catch (err) {
      alert("Klasör silinemedi: " + err.message);
    }
  }

  async function confirmDelete() {
    if (!pendingDelete) return;
    await deleteRecording(pendingDelete.id);
    setRecordings((prev) => prev.filter((r) => r.id !== pendingDelete.id));
    if (selected?.id === pendingDelete.id) setSelected(null);
    setPendingDelete(null);
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-brand">
          <span className="app-brand-mark" aria-hidden="true" />
          <h1 className="app-title">Ses Kayıt Paneli</h1>
        </div>

        <nav className="top-tabs" aria-label="Ana görünüm">
          <button
            className={`top-tab ${activeTab === "recordings" ? "is-active" : ""}`}
            onClick={() => setActiveTab("recordings")}
          >
            Kayıtlar
          </button>
          <button
            className={`top-tab ${activeTab === "live" ? "is-active" : ""}`}
            onClick={() => setActiveTab("live")}
          >
            Canlı İzleme
          </button>
        </nav>

        <div className="app-actions">
          <UploadButton onUploaded={(recording) => handleNewRecording(recording, { openDetails: true, isNew: true })} />
          <RecordButton onRecordingFinished={(recording) => handleNewRecording(recording, { openDetails: false, isNew: true })} />
        </div>
      </header>

      {activeTab === "recordings" ? (
        <main className="app-main">
          <aside className="sidebar" style={{ display: 'flex', flexDirection: 'column' }}>
            <div className="sidebar-heading mono" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>KAYITLAR · {recordings.length}</span>
              <button
                onClick={handleCreateFolder}
                style={{ background: 'var(--accent-record)', border: 'none', color: 'white', padding: '4px 8px', borderRadius: '4px', cursor: 'pointer', fontSize: '0.7rem' }}
              >
                + Klasör Ekle
              </button>
            </div>
            {loadError && <div className="list-status list-error">Liste yüklenemedi: {loadError}</div>}

            <RecordingList
              recordings={recordings}
              folders={folders}
              selectedId={selected?.id}
              onSelect={setSelected}
              onDelete={setPendingDelete}
              onEdit={handleEditRecording}
              isLoading={isLoading}
              onMoveRecording={handleMoveRecording}
              onDeleteFolder={handleDeleteFolder}
            />
          </aside>

          <section className="player-area">
            <WaveformPlayer
              recording={selected}
              onUpdateSuccess={handleUpdateRecordingState}
            />
          </section>
        </main>
      ) : (
        <main className="app-main live-main">
          <LiveMonitor />
        </main>
      )}

      {pendingDelete && (
        <ConfirmDialog
          title="Kaydı sil"
          message={`"${pendingDelete.title}" kalıcı olarak silinecek. Bu işlem geri alınamaz.`}
          onConfirm={confirmDelete}
          onCancel={() => setPendingDelete(null)}
        />
      )}

      {editingRecording && (
        <RecordingDetailsModal
          recording={editingRecording}
          folders={folders}
          isNew={isNewRecording}
          onSave={handleSaveDetails}
          onCancel={() => setEditingRecording(null)}
        />
      )}
    </div>
  );
}
