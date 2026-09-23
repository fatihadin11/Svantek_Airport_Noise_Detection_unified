import { useRef, useState } from "react";
import { uploadEncryptedRecording, uploadRecording } from "../api/recordings";

const AUDIO_EXTENSIONS = new Set([".wav", ".mp3", ".m4a", ".flac", ".ogg"]);

function getExtension(fileName) {
  const dotIndex = fileName.lastIndexOf(".");
  return dotIndex >= 0 ? fileName.slice(dotIndex).toLowerCase() : "";
}

export default function UploadButton({ onUploaded }) {
  const inputRef = useRef(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState(null);

  async function handleFileChange(event) {
    const files = Array.from(event.target.files || []);
    if (files.length === 0) return;

    const encryptedFiles = files.filter((file) => getExtension(file.name) === ".enc");
    const audioFile = files.find((file) => AUDIO_EXTENSIONS.has(getExtension(file.name)));
    const csvFiles = files.filter((file) => getExtension(file.name) === ".csv");
    const csvFile = csvFiles[0] || null;
    const encryptedAudio = encryptedFiles.find((file) => /\.wav\.enc$/i.test(file.name))
      || (encryptedFiles.length === 1 ? encryptedFiles[0] : null);
    const encryptedCsv = encryptedFiles.find((file) => /\.csv\.enc$/i.test(file.name)) || null;
    const encryptedSvl = encryptedFiles.find((file) => /\.svl\.enc$/i.test(file.name)) || null;

    if (!audioFile && !encryptedAudio) {
      setError("Bir ses dosyası veya audio.wav.enc seçin.");
      event.target.value = "";
      return;
    }

    if (csvFiles.length > 1) {
      setError("Tek kayıt için yalnızca bir CSV dosyası seçin.");
      event.target.value = "";
      return;
    }

    setError(null);
    setIsUploading(true);
    try {
      const recording = encryptedAudio
        ? await uploadEncryptedRecording(encryptedAudio, encryptedCsv, encryptedSvl)
        : await uploadRecording(audioFile, csvFile);
      // Manuel upload sonrası detay ekranı açılır; SVAN kaydı sonrası açılmaz.
      onUploaded(recording);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsUploading(false);
      event.target.value = ""; // aynı dosyanın tekrar seçilebilmesi için
    }
  }

  return (
    <div className="upload-control">
      <button
        className="upload-button"
        onClick={() => inputRef.current?.click()}
        disabled={isUploading}
        title="WAV + CSV veya Pi sisteminden audio.wav.enc (+ isteğe bağlı CSV/SVL .enc) seçebilirsin."
      >
        {isUploading ? "Yükleniyor…" : "Ses / Şifreli Kayıt Yükle"}
      </button>
      <input
        ref={inputRef}
        type="file"
        accept=".wav,.mp3,.m4a,.flac,.ogg,.csv,.enc"
        multiple
        onChange={handleFileChange}
        hidden
      />
      {error && <span className="record-error">{error}</span>}
    </div>
  );
}
