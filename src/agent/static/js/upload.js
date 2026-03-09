// Create HTML DOM elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const statusEl = document.getElementById('status');
const liveEl = document.getElementById('live');
const downloadLink = document.getElementById('download-link');

let es = null; // EventSource handle

// Add event listeners
dropZone.addEventListener('click', () => fileInput.click())
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
})
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    uploadFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => uploadFile(fileInput.files[0]))

function resetUI() {
    statusEl.textContent = "";
    liveEl.textContent = "";
    downloadLink.style.display = "none";
    downloadLink.href = "#";
    if (es) {
        es.close();
        es = null;
    }
}

async function uploadFile(file) {
  if (!file) return;

  resetUI();
  statusEl.textContent = `Uploading ${file.name}...`;

  const formData = new FormData();
  formData.append('file', file);

  try {
    // This must match with the audio upload route from 'agent/routers/transcription.py'
    const res = await fetch('/transcription/upload-audio', {
      method: 'POST',
      body: formData
    });

    let data = null;
    const ct = res.headers.get("content-type") || "";
    if (ct.includes("application/json")) data = await res.json();
    else data = { detail: await res.text() };

    if (!res.ok) {
      statusEl.textContent = `❌ Error: ${data.detail || 'Unknown error'}`;
      return;
    }

    // Expected from backend:
    // { job_id, stream_url, status_url, download_url }
    const { job_id, stream_url, download_url } = data;

    statusEl.textContent = `✅ Uploaded. Job: ${job_id}\nTranscribing...`;

    // Start SSE stream
    es = new EventSource(stream_url);

    es.onmessage = (evt) => {
      let msg;
      try {
        msg = JSON.parse(evt.data);
      } catch (e) {
        // If backend ever sends non-JSON, show it anyway
        liveEl.textContent += evt.data + "\n";
        return;
      }

      if (msg.type === "meta") {
        // Optional: show meta info
        const lang = msg.language ? `Language: ${msg.language}` : "";
        const dur = (msg.duration_s != null) ? `Duration: ${msg.duration_s}s` : "";
        const metaLine = [lang, dur].filter(Boolean).join(" | ");
        if (metaLine) statusEl.textContent = `✅ Uploaded. Job: ${job_id}\n${metaLine}\nTranscribing...`;
        return;
      }

      if (msg.type === "segment") {
        liveEl.textContent += (msg.text || "") + " ";
        // Auto-scroll for long transcripts
        window.scrollTo(0, document.body.scrollHeight);
        return;
      }

      if (msg.type === "done") {
        es.close();
        es = null;

        // If your done event includes full text, you can replace the live text:
        if (typeof msg.text === "string" && msg.text.length > 0) {
          liveEl.textContent = msg.text;
        }

        statusEl.textContent = `✅ Done. Job: ${job_id}`;
        downloadLink.href = download_url;
        downloadLink.style.display = "inline";
        return;
      }

      if (msg.type === "error") {
        es.close();
        es = null;
        statusEl.textContent = `❌ Transcription error: ${msg.message || 'Unknown error'}`;
        return;
      }
    };

    es.onerror = () => {
      // SSE connection dropped (worker down, server restarted, proxy issue, etc.)
      if (es) es.close();
      es = null;
      statusEl.textContent += `\n⚠️ Stream disconnected. You can refresh or check status endpoint.`;
    };

  } catch (err) {
    statusEl.textContent = `❌ Upload failed: ${err.message}`;
  }
}