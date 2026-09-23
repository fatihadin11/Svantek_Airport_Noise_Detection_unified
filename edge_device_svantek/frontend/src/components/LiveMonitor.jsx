import { useEffect, useRef, useState } from "react";
import { fetchLiveLatest, fetchLiveStatus, startLiveMeasurement, stopLiveMeasurement } from "../api/live";

const MAX_POINTS = 90;

function formatDb(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  return value.toFixed(1);
}

function LiveTrend({ samples, thresholdDb, showThreshold }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(320, Math.floor(rect.width * ratio));
    const height = Math.max(160, Math.floor(rect.height * ratio));

    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, width, height);

    ctx.globalAlpha = 0.9;
    ctx.lineWidth = 1 * ratio;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";

    const padding = 22 * ratio;
    const graphW = width - padding * 2;
    const graphH = height - padding * 2;

    for (let i = 0; i <= 4; i += 1) {
      const y = padding + (graphH * i) / 4;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }

    const values = samples.map((item) => item.spl).filter((v) => typeof v === "number");
    if (values.length < 2) return;

    const thresholdValue = showThreshold ? Number(thresholdDb) : null;
    const scaleValues = thresholdValue !== null ? [...values, thresholdValue] : values;
    const min = Math.min(...scaleValues, 30);
    const max = Math.max(...scaleValues, 90);
    const span = Math.max(10, max - min);

    if (thresholdValue !== null) {
      const y = padding + graphH - ((thresholdValue - min) / span) * graphH;
      ctx.strokeStyle = "rgba(239, 68, 68, 0.9)";
      ctx.lineWidth = 1.5 * ratio;
      ctx.setLineDash([6 * ratio, 5 * ratio]);
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(239, 68, 68, 0.95)";
      ctx.font = `${11 * ratio}px monospace`;
      ctx.fillText(`Eşik ${thresholdValue.toFixed(0)} dB`, padding + 4 * ratio, Math.max(14 * ratio, y - 5 * ratio));
    }

    ctx.strokeStyle = "rgba(242, 166, 90, 0.95)";
    ctx.lineWidth = 2 * ratio;
    ctx.beginPath();

    values.forEach((value, index) => {
      const x = padding + (graphW * index) / Math.max(1, values.length - 1);
      const y = padding + graphH - ((value - min) / span) * graphH;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });

    ctx.stroke();

    ctx.fillStyle = "rgba(255,255,255,0.45)";
    ctx.font = `${11 * ratio}px monospace`;
    ctx.fillText(`${max.toFixed(0)} dB`, 4 * ratio, padding + 4 * ratio);
    ctx.fillText(`${min.toFixed(0)} dB`, 4 * ratio, height - padding + 4 * ratio);
  }, [samples, thresholdDb, showThreshold]);

  return (
    <div className="live-trend-wrap">
      <canvas ref={canvasRef} className="live-trend-canvas" />
      {samples.length === 0 && (
        <div className="live-empty">Canlı izleme başlatılınca ses seviyesi grafiği burada görünecek.</div>
      )}
    </div>
  );
}

export default function LiveMonitor() {
  const [status, setStatus] = useState(null);
  const [latest, setLatest] = useState(null);
  const [samples, setSamples] = useState([]);
  const [intervalMs, setIntervalMs] = useState(1000);
  const [isLiveActive, setIsLiveActive] = useState(false);
  const [isBusy, setIsBusy] = useState(false);
  const [error, setError] = useState(null);

  const [thresholdEnabled, setThresholdEnabled] = useState(false);
  const [thresholdDb, setThresholdDb] = useState(120);
  const [triggerHoldSec, setTriggerHoldSec] = useState(1);
  const [releaseHoldSec, setReleaseHoldSec] = useState(5);
  const [autoRearm, setAutoRearm] = useState(false);
  const [rearmCooldownSec, setRearmCooldownSec] = useState(30);
  const [setupName, setSetupName] = useState("AUTO120");
  const [hasTriggered, setHasTriggered] = useState(false);
  const [notice, setNotice] = useState(null);

  const latestInFlightRef = useRef(false);
  const stoppingRef = useRef(false);

  async function refreshStatus() {
    const data = await fetchLiveStatus();
    setStatus(data);

    const session = data?.live_session;
    const active = Boolean(session?.active || data?.active);
    setIsLiveActive(active);

    if (["threshold_trigger", "threshold_rearm", "threshold_complete"].includes(session?.mode)) {
      setThresholdEnabled(true);
      setHasTriggered(Boolean(session.triggered));
      setAutoRearm(Boolean(session.auto_rearm));
      if (typeof session.threshold_db === "number") setThresholdDb(session.threshold_db);
      if (typeof session.trigger_hold_sec === "number") setTriggerHoldSec(session.trigger_hold_sec);
      if (typeof session.release_hold_sec === "number") setReleaseHoldSec(session.release_hold_sec);
      if (typeof session.rearm_cooldown_sec === "number") setRearmCooldownSec(session.rearm_cooldown_sec);
      if (session.setup_name) setSetupName(session.setup_name);
    }

    return data;
  }

  async function refreshLatest() {
    if (latestInFlightRef.current || stoppingRef.current) return null;

    latestInFlightRef.current = true;
    try {
      const data = await fetchLiveLatest();
      setLatest(data);

      const live = data?.live;
      if (live?.available && typeof live.spl === "number") {
        setSamples((prev) => {
          const next = [
            ...prev,
            {
              at: Date.now(),
              spl: live.spl,
              leq: live.leq,
              lpeak: live.lpeak,
            },
          ];
          return next.slice(-MAX_POINTS);
        });
      }

      // Uzun/kümülatif Leq ile karar verme. Edge agent kısa süreli anlık
      // SPL değerini ve süre koşulunu değerlendirir; frontend yalnızca sonucu gösterir.
      const edgeSession = data?.live_session;

      if (edgeSession?.mode === "threshold_trigger") {
        setIsLiveActive(Boolean(edgeSession.active));
        setHasTriggered(Boolean(edgeSession.triggered));
        setAutoRearm(Boolean(edgeSession.auto_rearm));

        if (edgeSession.rearmed_at) {
          setNotice(
            `${Number(edgeSession.completed_count || 0)} kayıt tamamlandı. ` +
            `Eşik izleme ${Number(edgeSession.cycle_index || 1)}. tur için yeniden başladı.`
          );
        }
      }

      if (edgeSession?.mode === "threshold_rearm") {
        setIsLiveActive(true);
        setHasTriggered(false);
        setAutoRearm(true);

        const remaining = Number(edgeSession.cooldown_remaining_sec || 0);
        setNotice(
          `Kayıt backend tarafından alındı. ` +
          `${Math.ceil(remaining)} saniye sonra eşik izleme yeniden başlayacak.`
        );
      }

      if (
        edgeSession?.mode === "threshold_complete" &&
        edgeSession?.state === "uploaded"
      ) {
        setHasTriggered(true);
        setIsLiveActive(false);
        setNotice("Kayıt otomatik kapatıldı ve sunucuya aktarıldı.");
      }

      if (edgeSession?.state === "error") {
        setIsLiveActive(false);
        setError(
          edgeSession.auto_finalize_error ||
          edgeSession.rearm_error ||
          "Otomatik kayıt döngüsünde hata oluştu."
        );
      }

      return data;
    } finally {
      latestInFlightRef.current = false;
    }
  }

  useEffect(() => {
    refreshStatus().catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    if (!isLiveActive) return undefined;

    let cancelled = false;

    async function tick() {
      try {
        if (!cancelled) {
          await refreshLatest();
          setError(null);
        }
      } catch (err) {
        if (!cancelled) setError(err.message);
      }
    }

    tick();
    const id = setInterval(tick, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [isLiveActive, intervalMs, thresholdEnabled, thresholdDb]);

  async function handleStart() {
    setIsBusy(true);
    setError(null);
    setNotice(null);
    setHasTriggered(false);

    try {
      const numericThreshold = Number(thresholdDb);
      const numericHoldSec = Number(triggerHoldSec);
      const numericReleaseSec = Number(releaseHoldSec);
      const numericRearmCooldown = Number(rearmCooldownSec);

      if (thresholdEnabled && (!Number.isFinite(numericThreshold) || numericThreshold < 24 || numericThreshold > 136)) {
        throw new Error("Eşik değeri 24 ile 136 dB arasında olmalıdır.");
      }

      if (thresholdEnabled && (!Number.isFinite(numericHoldSec) || numericHoldSec < 0.5 || numericHoldSec > 30)) {
        throw new Error("Tetikleme süresi 0.5 ile 30 saniye arasında olmalıdır.");
      }

      if (thresholdEnabled && (!Number.isFinite(numericReleaseSec) || numericReleaseSec < 0.5 || numericReleaseSec > 300)) {
        throw new Error("Otomatik kapanış süresi 0.5 ile 300 saniye arasında olmalıdır.");
      }

      if (thresholdEnabled && autoRearm && (!Number.isFinite(numericRearmCooldown) || numericRearmCooldown < 5 || numericRearmCooldown > 900)) {
        throw new Error("Yeniden başlatma cooldown süresi 5 ile 900 saniye arasında olmalıdır.");
      }

      await startLiveMeasurement({
        threshold_enabled: thresholdEnabled,
        threshold_db: numericThreshold,
        trigger_hold_sec: numericHoldSec,
        release_hold_sec: numericReleaseSec,
        auto_rearm: thresholdEnabled && autoRearm,
        rearm_cooldown_sec: numericRearmCooldown,
        setup_name: setupName.trim(),
        folder: "Otomatik",
        tag: thresholdEnabled ? `threshold-${numericThreshold}db` : "",
        color: thresholdEnabled ? "#ef4444" : "#f2a65a",
        title: thresholdEnabled ? `Otomatik ${numericThreshold} dB` : null,
      });

      setSamples([]);
      await refreshStatus();
      await refreshLatest();
      setIsLiveActive(true);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsBusy(false);
    }
  }

  async function handleStop() {
    stoppingRef.current = true;
    setIsLiveActive(false);
    setIsBusy(true);
    setError(null);

    try {
      const result = await stopLiveMeasurement({
        finalize: thresholdEnabled && hasTriggered,
      });
      setStatus(result);

      if (result?.stopped === false || result?.status === "stop_requested_but_still_running") {
        setError("Canlı izleme durdurulamadı. Cihazı kontrol edip tekrar deneyin.");
      } else {
        setLatest(null);
        await refreshStatus();
      }
    } catch (err) {
      setError(err.message);
      setIsLiveActive(true);
    } finally {
      stoppingRef.current = false;
      setIsBusy(false);
    }
  }

  const live = latest?.live || {};
  const triggerSession = latest?.live_session || status?.live_session || {};
  const sessionMode = triggerSession?.mode;
  const thresholdModeActive = isLiveActive && sessionMode === "threshold_trigger";
  const rearmModeActive = isLiveActive && sessionMode === "threshold_rearm";
  const thresholdWorkflowActive = thresholdModeActive || rearmModeActive;
  const triggerElapsed = Number(triggerSession?.over_threshold_elapsed_sec || 0);
  const releaseElapsed = Number(triggerSession?.below_threshold_elapsed_sec || 0);
  const currentTriggerValue = triggerSession?.current_trigger_value_db;
  const isBelowThreshold =
    typeof currentTriggerValue === "number" &&
    currentTriggerValue < Number(thresholdDb);
  const edgeFinalizing = triggerSession?.state === "finalizing";
  const edgeRearming = triggerSession?.state === "rearming";
  const rearmRemaining = Number(triggerSession?.cooldown_remaining_sec || 0);
  const completedCount = Number(triggerSession?.completed_count || 0);
  const uploadCompleted =
    triggerSession?.mode === "threshold_complete" &&
    triggerSession?.state === "uploaded";

  let stateTitle = "Kapalı";
  let stateDetail = "Beklemede";

  if (uploadCompleted) {
    stateTitle = "Aktarıldı";
    stateDetail = "Otomatik kayıt başarıyla tamamlandı";
  } else if (edgeFinalizing) {
    stateTitle = "Kapatılıyor";
    stateDetail = "Dosyalar hazırlanıyor ve sunucuya aktarılıyor";
  } else if (edgeRearming) {
    stateTitle = "Yeniden başlatılıyor";
    stateDetail = "SVAN setup ve logger yeniden hazırlanıyor";
  } else if (rearmModeActive) {
    stateTitle = "Cooldown";
    stateDetail = `${rearmRemaining.toFixed(1)} saniye sonra yeniden eşik izleme`;
  } else if (isLiveActive && !thresholdWorkflowActive) {
    stateTitle = "Canlı";
    stateDetail = "Yalnızca ölçüm izleniyor";
  } else if (thresholdModeActive && !hasTriggered) {
    stateTitle = "Eşik bekleniyor";
    stateDetail = `Anlık SPL · ${Number(thresholdDb).toFixed(0)} dB · ${Number(triggerHoldSec).toFixed(1)} s`;
  } else if (thresholdModeActive && hasTriggered && isBelowThreshold) {
    stateTitle = "Otomatik kapanış";
    stateDetail = `${releaseElapsed.toFixed(1)} / ${Number(releaseHoldSec).toFixed(1)} saniye`;
  } else if (thresholdModeActive && hasTriggered) {
    stateTitle = "Kayıt aktif";
    stateDetail = `Kısa süreli SPL: ${formatDb(currentTriggerValue)} dB`;
  }

  let stopButtonLabel = "Canlı Durdur";
  if (edgeFinalizing) {
    stopButtonLabel = "Otomatik kapatılıyor ve aktarılıyor…";
  } else if (edgeRearming) {
    stopButtonLabel = "Yeniden eşik izleme başlatılıyor…";
  } else if (rearmModeActive) {
    stopButtonLabel = "Döngüyü Durdur";
  } else if (isBusy) {
    stopButtonLabel = thresholdEnabled && hasTriggered
      ? "Durduruluyor ve aktarılıyor…"
      : "Durduruluyor…";
  } else if (thresholdEnabled && hasTriggered) {
    stopButtonLabel = "Durdur ve Aktar";
  } else if (thresholdEnabled) {
    stopButtonLabel = "Eşik İzlemeyi İptal Et";
  }

  return (
    <section className="live-panel">
      <div className="live-header">
        <div>
          <h2>Canlı İzleme</h2>
          <p>
            Anlık ses seviyesini izleyin. Otomatik eşik kaydı, sekiz saatlik
            kümülatif Leq yerine kısa süreli SPL seviyesini kullanır.
          </p>
        </div>

        <div className="live-actions">
          <label className="live-select-label">
            Okuma aralığı
            <select value={intervalMs} onChange={(e) => setIntervalMs(Number(e.target.value))} disabled={isLiveActive}>
              <option value={1000}>1 saniye</option>
              <option value={500}>0.5 saniye</option>
            </select>
          </label>

          {!isLiveActive ? (
            <button className="live-primary" onClick={handleStart} disabled={isBusy}>
              {isBusy ? "Başlatılıyor…" : thresholdEnabled ? "Eşik İzlemeyi Başlat" : "Canlı Başlat"}
            </button>
          ) : (
            <button
              className="live-danger"
              onClick={handleStop}
              disabled={isBusy || edgeFinalizing || edgeRearming}
            >
              {stopButtonLabel}
            </button>
          )}
        </div>
      </div>

      <div className={`trigger-config ${thresholdEnabled ? "is-enabled" : ""}`}>
        <div className="trigger-config-heading">
          <div>
            <strong>Otomatik eşik kaydı</strong>
            <span>Canlı izleme ile aynı ekranda çalışır.</span>
          </div>

          <label className="trigger-switch">
            <input
              type="checkbox"
              checked={thresholdEnabled}
              onChange={(e) => setThresholdEnabled(e.target.checked)}
              disabled={isLiveActive}
            />
            <span>{thresholdEnabled ? "Açık" : "Kapalı"}</span>
          </label>
        </div>

        {thresholdEnabled && (
          <>
            <div className="trigger-fields">
            <label>
              Eşik
              <div className="trigger-number-wrap">
                <input
                  type="number"
                  min="24"
                  max="136"
                  step="1"
                  value={thresholdDb}
                  onChange={(e) => setThresholdDb(e.target.value)}
                  disabled={isLiveActive}
                />
                <span>dB</span>
              </div>
            </label>

            <label>
              Eşik üstünde kalma
              <div className="trigger-number-wrap">
                <input
                  type="number"
                  min="0.5"
                  max="30"
                  step="0.5"
                  value={triggerHoldSec}
                  onChange={(e) => setTriggerHoldSec(e.target.value)}
                  disabled={isLiveActive}
                />
                <span>sn</span>
              </div>
            </label>

            <label>
              Eşik altında kalma
              <div className="trigger-number-wrap">
                <input
                  type="number"
                  min="0.5"
                  max="300"
                  step="0.5"
                  value={releaseHoldSec}
                  onChange={(e) => setReleaseHoldSec(e.target.value)}
                  disabled={isLiveActive}
                />
                <span>sn</span>
              </div>
            </label>

            <label>
              SVAN setup adı
              <input
                type="text"
                maxLength="8"
                value={setupName}
                onChange={(e) => setSetupName(e.target.value.toUpperCase())}
                disabled={isLiveActive}
              />
            </label>

            <div className="trigger-info">
              <strong>Kısa süreli tetikleme</strong>
              <span>
                Anlık SPL {Number(thresholdDb || 0).toFixed(0)} dB üzerinde
                {` ${Number(triggerHoldSec || 0).toFixed(1)} saniye`} kalırsa kayıt tetiklenir.
                Tetiklemeden sonra eşik altında
                {` ${Number(releaseHoldSec || 0).toFixed(1)} saniye`} kalırsa cihaz otomatik
                durdurulur ve dosyalar sunucuya aktarılır.
              </span>
            </div>
          </div>

          <div className={`trigger-loop-config ${autoRearm ? "is-enabled" : ""}`}>
            <div className="trigger-loop-copy">
              <strong>Olay sonrası tekrar eşik izle</strong>
              <span>
                Kayıt backend tarafından alındıktan sonra cooldown uygulanır ve
                eşik izleme yeni bir logger oturumuyla yeniden başlar.
              </span>
            </div>

            <label className="trigger-switch">
              <input
                type="checkbox"
                checked={autoRearm}
                onChange={(e) => setAutoRearm(e.target.checked)}
                disabled={isLiveActive}
              />
              <span>{autoRearm ? "Açık" : "Kapalı"}</span>
            </label>

            {autoRearm && (
              <label className="trigger-loop-cooldown">
                Cooldown
                <div className="trigger-number-wrap">
                  <input
                    type="number"
                    min="5"
                    max="900"
                    step="5"
                    value={rearmCooldownSec}
                    onChange={(e) => setRearmCooldownSec(e.target.value)}
                    disabled={isLiveActive}
                  />
                  <span>sn</span>
                </div>
              </label>
            )}
          </div>
          </>
        )}
      </div>

      {thresholdWorkflowActive && (
        <div className={`trigger-state ${hasTriggered ? "is-triggered" : ""}`}>
          <strong>
            {rearmModeActive
              ? "Kayıt aktarıldı — yeniden başlatma bekleniyor"
              : hasTriggered
                ? "Eşik aşıldı — otomatik kayıt aktif"
                : "Eşik bekleniyor"}
          </strong>
          <span>
            {edgeRearming
              ? "SVAN setup, logger ve ölçüm yeniden başlatılıyor."
              : rearmModeActive
                ? `${completedCount} kayıt tamamlandı · ${rearmRemaining.toFixed(1)} sn sonra yeniden eşik izleme`
                : edgeFinalizing
                  ? "SVAN durduruluyor; WAV, CSV ve SVL hazırlanıp sunucuya aktarılıyor."
                  : hasTriggered && isBelowThreshold
                    ? `${formatDb(currentTriggerValue)} dB · eşik altında ${releaseElapsed.toFixed(1)} / ${Number(releaseHoldSec).toFixed(1)} sn`
                    : hasTriggered
                      ? `${formatDb(currentTriggerValue)} dB · kayıt aktif; sesin eşik altına düşmesi bekleniyor.`
                      : `${formatDb(currentTriggerValue)} dB · eşik üstünde ${triggerElapsed.toFixed(1)} / ${Number(triggerHoldSec).toFixed(1)} sn`}
          </span>
        </div>
      )}

      {error && <div className="live-error">{error}</div>}
      {notice && <div className="live-warning">{notice}</div>}

      <div className="live-cards">
        <div className="live-db-card">
          <span className="live-card-label">Anlık SPL</span>
          <strong>{formatDb(live.spl)}</strong>
          <span>dB</span>
        </div>

        <div className="live-small-card">
          <span>Leq</span>
          <strong>{formatDb(live.leq)}</strong>
        </div>

        <div className="live-small-card">
          <span>Lpeak</span>
          <strong>{formatDb(live.lpeak)}</strong>
        </div>

        <div className="live-small-card">
          <span>Lmax / Lmin</span>
          <strong>{formatDb(live.lmax)} / {formatDb(live.lmin)}</strong>
        </div>

        <div className="live-small-card">
          <span>Durum</span>
          <strong>{stateTitle}</strong>
          <small>{stateDetail}</small>
        </div>

        <div className="live-small-card">
          <span>Overload</span>
          <strong>{live.overload ? "Var" : "Yok"}</strong>
          <small>Underrange: {live.underrange ?? "—"}</small>
        </div>
      </div>

      <LiveTrend
        samples={samples}
        thresholdDb={Number(thresholdDb)}
        showThreshold={thresholdEnabled}
      />
    </section>
  );
}
