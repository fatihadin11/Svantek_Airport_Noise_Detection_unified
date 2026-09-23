import { useMemo, useState } from "react";
import { runPassiveLiveTest } from "../api/recordings";
import "./LivePassiveTestPanel.css";

const DEFAULT_CMD = "#2,i,1,S?,R?,P?,M?,N?,T?,V?,v?;";

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(1);
}

export default function LivePassiveTestPanel() {
  const [command, setCommand] = useState(DEFAULT_CMD);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isRunning, setIsRunning] = useState(false);

  const verdict = useMemo(() => {
    if (!result) return null;
    if (result.active_session) {
      return {
        type: "warning",
        title: "Kayıt aktifken pasif test kapalı",
        text: "Edge üzerinde aktif kayıt var. Canlı test kayıt akışına karışmasın diye çalıştırılmadı.",
      };
    }
    if (result.live?.available) {
      return {
        type: "success",
        title: "Pasif canlı okuma mümkün görünüyor",
        text: "Cihazı start/record komutu göndermeden #2 ölçüm sonucu döndü. Bu, web panelde cihaz ekranına benzer dB göstergesi için iyi işaret.",
      };
    }
    return {
      type: "warning",
      title: "#2 pasif sonuç dönmedi",
      text: "Cihaz açık olsa bile USB API ölçüm sonucu vermedi. Canlı izleme için measurement başlatma veya logger kapalı ölçüm modu test edilecek.",
    };
  }, [result]);

  async function handleRunTest() {
    setIsRunning(true);
    setError(null);
    try {
      const data = await runPassiveLiveTest(command);
      setResult(data);
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setIsRunning(false);
    }
  }

  const live = result?.live || {};

  return (
    <section className="live-passive-panel">
      <div className="live-passive-header">
        <div>
          <p className="live-eyebrow mono">SVAN 971 USB TEST</p>
          <h2>Pasif Canlı dB Okuma Testi</h2>
          <p>
            Bu ekran kayıt başlatmaz. Amaç, cihaz sadece açıkken ve ekranda dB değeri görünürken USB üzerinden
            <code>#2</code> sonucunu alabiliyor muyuz onu doğrulamak.
          </p>
        </div>
        <button className="live-test-button" onClick={handleRunTest} disabled={isRunning}>
          {isRunning ? "Test ediliyor…" : "Pasif #2 Testi Yap"}
        </button>
      </div>

      <div className="live-command-box">
        <label>
          Test komutu
          <input value={command} onChange={(event) => setCommand(event.target.value)} spellCheck={false} />
        </label>
        <p>
          Not: Bu komut sadece okuma yapar. <code>#1,S1</code> gibi kayıt/measurement başlatma komutu göndermez.
        </p>
      </div>

      {error && <div className="live-status live-error">Test hatası: {error}</div>}

      {verdict && (
        <div className={`live-status live-${verdict.type}`}>
          <strong>{verdict.title}</strong>
          <span>{verdict.text}</span>
        </div>
      )}

      <div className="live-metric-grid">
        <div className="live-metric-card live-metric-main">
          <span>Anlık SPL / L</span>
          <strong>{formatNumber(live.spl)} dB</strong>
        </div>
        <div className="live-metric-card">
          <span>Leq</span>
          <strong>{formatNumber(live.leq)} dB</strong>
        </div>
        <div className="live-metric-card">
          <span>Lpeak</span>
          <strong>{formatNumber(live.lpeak)} dB</strong>
        </div>
        <div className="live-metric-card">
          <span>Lmax / Lmin</span>
          <strong>{formatNumber(live.lmax)} / {formatNumber(live.lmin)}</strong>
        </div>
        <div className="live-metric-card">
          <span>Overload</span>
          <strong>{live.overload === undefined ? "—" : live.overload ? "Var" : "Yok"}</strong>
        </div>
        <div className="live-metric-card">
          <span>Cihaz durumu</span>
          <strong>{result?.status?.measurement_state || "—"}</strong>
        </div>
      </div>

      {result && (
        <div className="live-raw-grid">
          <div>
            <h3>Ham cevaplar</h3>
            <pre>{JSON.stringify({ status_raw: result.status?.raw, live_raw: result.live?.raw }, null, 2)}</pre>
          </div>
          <div>
            <h3>Tüm JSON</h3>
            <pre>{JSON.stringify(result, null, 2)}</pre>
          </div>
        </div>
      )}
    </section>
  );
}
