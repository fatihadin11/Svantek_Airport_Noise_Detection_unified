async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });

  let data = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }

  if (!response.ok) {
    const message = data?.detail || data?.error || response.statusText || "İstek başarısız.";
    throw new Error(typeof message === "string" ? message : JSON.stringify(message));
  }

  return data;
}

export function fetchLiveStatus() {
  return requestJson("/api/live/status");
}

export function startLiveMeasurement(config = {}) {
  return requestJson("/api/live/start", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

export function stopLiveMeasurement({ finalize = false } = {}) {
  return requestJson("/api/live/stop", {
    method: "POST",
    body: JSON.stringify({ finalize }),
  });
}

export function fetchLiveLatest() {
  return requestJson("/api/live/latest");
}
