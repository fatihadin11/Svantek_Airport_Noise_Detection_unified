const BASE_URL = "/api";

async function handleResponse(res) {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch { }
    throw new Error(detail);
  }
  return res.json();
}

export async function fetchRecordings() {
  const res = await fetch(`${BASE_URL}/recordings`);
  return handleResponse(res);
}

export async function fetchWaveform(id) {
  const res = await fetch(`${BASE_URL}/recordings/${id}/waveform`);
  return handleResponse(res);
}

export async function fetchRecordingAnalysis(id) {
  const res = await fetch(`${BASE_URL}/recordings/${id}/analysis`);
  return handleResponse(res);
}

export async function startRecordingAnalysis(id) {
  const res = await fetch(`${BASE_URL}/recordings/${id}/analysis`, { method: "POST" });
  return handleResponse(res);
}

export async function fetchCsvGraph(id) {
  const res = await fetch(`${BASE_URL}/recordings/${id}/csv-graph`);
  return handleResponse(res);
}

export async function fetchCsvViews(id) {
  const res = await fetch(`${BASE_URL}/recordings/${id}/csv-views`);
  return handleResponse(res);
}

export async function fetchCsvViewGraph(id, { view = "logger_octave", metric = "Leq" } = {}) {
  const params = new URLSearchParams();
  params.set("view", view);
  if (metric) params.set("metric", metric);
  const res = await fetch(`${BASE_URL}/recordings/${id}/csv-view-graph?${params.toString()}`);
  return handleResponse(res);
}

export async function startRecording(metadata = {}) {
  const res = await fetch(`${BASE_URL}/recording-session/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(metadata),
  });
  return handleResponse(res);
}

export async function stopRecording() {
  const res = await fetch(`${BASE_URL}/recording-session/stop`, { method: "POST" });
  return handleResponse(res);
}

export async function runPassiveLiveTest(command = null) {
  const params = new URLSearchParams();
  if (command) params.set("cmd", command);
  const suffix = params.toString() ? `?${params.toString()}` : "";
  const res = await fetch(`${BASE_URL}/live/passive-test${suffix}`);
  return handleResponse(res);
}

// Eski mikrofon kayıt endpointleri debug için korunuyor.
export async function startLocalMicRecording() {
  const res = await fetch(`${BASE_URL}/recordings/start`, { method: "POST" });
  return handleResponse(res);
}

export async function stopLocalMicRecording() {
  const res = await fetch(`${BASE_URL}/recordings/stop`, { method: "POST" });
  return handleResponse(res);
}

export async function uploadRecording(file, csvFile = null) {
  const formData = new FormData();
  formData.append("file", file);
  if (csvFile) formData.append("csv_file", csvFile);

  const res = await fetch(`${BASE_URL}/recordings/upload`, { method: "POST", body: formData });
  return handleResponse(res);
}

export async function uploadEncryptedRecording(audioFile, csvFile = null, svlFile = null) {
  const formData = new FormData();
  formData.append("audio_file_enc", audioFile);
  if (csvFile) formData.append("csv_file_enc", csvFile);
  if (svlFile) formData.append("svl_file_enc", svlFile);
  const res = await fetch(`${BASE_URL}/recordings/upload-encrypted`, { method: "POST", body: formData });
  return handleResponse(res);
}

export async function deleteRecording(id) {
  const res = await fetch(`${BASE_URL}/recordings/${id}`, { method: "DELETE" });
  return handleResponse(res);
}

export async function updateRecording(id, data) {
  const res = await fetch(`${BASE_URL}/recordings/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return handleResponse(res);
}

export function audioUrl(id) {
  return `${BASE_URL}/recordings/${id}/audio`;
}

export async function fetchFolders() {
  const res = await fetch(`${BASE_URL}/recordings/folders/list`);
  return handleResponse(res);
}

export async function createFolder(name) {
  const res = await fetch(`${BASE_URL}/recordings/folders/create`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  return handleResponse(res);
}

export async function deleteFolder(name) {
  const res = await fetch(`${BASE_URL}/recordings/folders/${encodeURIComponent(name)}`, { method: "DELETE" });
  return handleResponse(res);
}

export async function deletePlainFiles(id) {
  const res = await fetch(`${BASE_URL}/encryption/jobs/${id}/delete-plain`, { method: "POST" });
  return handleResponse(res);
}
