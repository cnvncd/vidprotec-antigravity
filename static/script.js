/**
 * StealthMask — Frontend Logic
 * Handles file uploads, gallery, processing, progress, comparison, and downloads.
 */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
    jobId: null,
    files: [],           // [{original_name, stored_name, type, status, progress, localUrl}]
    processing: false,
    pollTimer: null,
};

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const $ = (sel) => document.querySelector(sel);

const dropzone        = $("#dropzone");
const fileInput       = $("#file-input");
const gallery         = $("#gallery");
const gallerySection  = $("#gallery-section");
const controlsSection = $("#controls-section");
const processBtn      = $("#process-btn");
const progressSection = $("#progress-section");
const progressList    = $("#progress-list");
const overallBar      = $("#overall-bar");
const overallPct      = $("#overall-pct");
const resultsSection  = $("#results-section");
const resultsGrid     = $("#results-grid");
const downloadAllBtn  = $("#download-all-btn");

// ---------------------------------------------------------------------------
// Drag & Drop + File Input
// ---------------------------------------------------------------------------
const ACCEPTED = new Set([
    "image/jpeg", "image/png", "image/webp",
    "video/mp4", "video/quicktime", "video/x-msvideo", "video/avi",
]);

dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
});
dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("drag-over");
});
dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
    handleFiles(e.dataTransfer.files);
});
fileInput.addEventListener("change", () => {
    handleFiles(fileInput.files);
    fileInput.value = "";
});

async function handleFiles(fileList) {
    const valid = [...fileList].filter((f) => {
        // Fallback: check extension if MIME is empty
        const ext = f.name.split(".").pop().toLowerCase();
        const extSet = new Set(["jpg", "jpeg", "png", "webp", "mp4", "mov", "avi"]);
        return ACCEPTED.has(f.type) || extSet.has(ext);
    });
    if (!valid.length) {
        showToast("Поддерживаются только JPG, PNG, WEBP, MP4, MOV, AVI", "warn");
        return;
    }

    const fd = new FormData();
    valid.forEach((f) => fd.append("files", f));

    try {
        const res = await fetch("/api/upload", { method: "POST", body: fd });
        const data = await res.json();
        if (!res.ok) { showToast(data.error, "error"); return; }

        state.jobId = data.job_id;

        // Merge local blob URLs for preview
        data.files.forEach((fm, i) => {
            const localFile = valid.find((v) => v.name === fm.original_name);
            fm.localUrl = localFile ? URL.createObjectURL(localFile) : null;
            state.files.push(fm);
        });

        renderGallery();
        show(gallerySection);
        show(controlsSection);
    } catch (err) {
        showToast("Ошибка загрузки файлов", "error");
    }
}

// ---------------------------------------------------------------------------
// Gallery
// ---------------------------------------------------------------------------
function renderGallery() {
    gallery.innerHTML = "";
    state.files.forEach((f, idx) => {
        const el = document.createElement("div");
        el.className = "gallery-item";

        const isVideo = f.type === "video";
        const media = isVideo
            ? `<video src="${f.localUrl}" muted playsinline></video>`
            : `<img src="${f.localUrl}" alt="${f.original_name}">`;

        el.innerHTML = `
            ${media}
            <div class="gallery-remove" data-idx="${idx}" title="Удалить">✕</div>
            <div class="px-2 py-1.5">
                <p class="text-xs truncate text-gray-400" title="${f.original_name}">${f.original_name}</p>
                <span class="badge badge-${f.status} mt-1">${statusLabel(f.status)}</span>
            </div>
        `;
        gallery.appendChild(el);
    });

    // Remove buttons
    gallery.querySelectorAll(".gallery-remove").forEach((btn) => {
        btn.addEventListener("click", (e) => {
            e.stopPropagation();
            removeFile(+btn.dataset.idx);
        });
    });
}

async function removeFile(idx) {
    const f = state.files[idx];
    if (!f) return;
    // Delete from server
    await fetch(`/api/delete-file/${state.jobId}/${f.stored_name}`, { method: "DELETE" }).catch(() => {});
    state.files.splice(idx, 1);
    renderGallery();
    if (!state.files.length) {
        hide(gallerySection);
        hide(controlsSection);
    }
}

function statusLabel(s) {
    return { pending: "Ожидание", processing: "Обработка…", done: "Готово", error: "Ошибка" }[s] || s;
}

// ---------------------------------------------------------------------------
// Process
// ---------------------------------------------------------------------------
processBtn.addEventListener("click", startProcessing);

async function startProcessing() {
    if (!state.jobId || state.processing) return;
    state.processing = true;
    processBtn.disabled = true;

    try {
        const res = await fetch(`/api/process/${state.jobId}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: "{}",
        });
        if (!res.ok) {
            const d = await res.json();
            showToast(d.error || "Ошибка запуска", "error");
            state.processing = false;
            processBtn.disabled = false;
            return;
        }

        show(progressSection);
        renderProgressList();
        startPolling();
    } catch {
        showToast("Ошибка соединения", "error");
        state.processing = false;
        processBtn.disabled = false;
    }
}

// ---------------------------------------------------------------------------
// Progress polling
// ---------------------------------------------------------------------------
function startPolling() {
    state.pollTimer = setInterval(pollStatus, 800);
}

async function pollStatus() {
    if (!state.jobId) return;
    try {
        const res = await fetch(`/api/status/${state.jobId}`);
        const data = await res.json();

        // Update file statuses
        data.files.forEach((sf, i) => {
            if (state.files[i]) {
                state.files[i].status = sf.status;
                state.files[i].progress = sf.progress;
            }
        });

        // Render progress
        renderProgressList();
        overallBar.style.width = data.progress + "%";
        overallPct.textContent = data.progress + "%";

        if (data.status === "done") {
            clearInterval(state.pollTimer);
            state.processing = false;
            showToast("✅ Обработка завершена!", "success");
            renderGallery();
            showResults();
        }
    } catch {
        // Network blip — keep polling
    }
}

function renderProgressList() {
    progressList.innerHTML = "";
    state.files.forEach((f) => {
        const pct = f.progress || 0;
        const div = document.createElement("div");
        div.className = "flex items-center gap-3 mb-2";
        div.innerHTML = `
            <span class="text-xs text-gray-400 w-40 truncate" title="${f.original_name}">${f.original_name}</span>
            <div class="progress-track flex-1">
                <div class="progress-fill" style="width:${pct}%"></div>
            </div>
            <span class="badge badge-${f.status} shrink-0">${statusLabel(f.status)}</span>
        `;
        progressList.appendChild(div);
    });
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------
function showResults() {
    show(resultsSection);
    resultsGrid.innerHTML = "";

    state.files.forEach((f) => {
        if (f.status !== "done") return;
        const outName = nameToMasked(f.original_name);
        const isVideo = f.type === "video";

        const card = document.createElement("div");
        card.className = "glass p-4 space-y-3";

        // Build compare widget
        const compareId = "compare-" + f.stored_name;
        if (isVideo) {
            card.innerHTML = `
                <p class="text-sm font-semibold text-gray-300">${f.original_name}</p>
                <div class="grid grid-cols-2 gap-2">
                    <div>
                        <span class="compare-label text-xs mb-1 inline-block">Оригинал</span>
                        <video src="/api/preview/${state.jobId}/${f.stored_name}" controls muted class="w-full rounded-lg border border-gray-700"></video>
                    </div>
                    <div>
                        <span class="compare-label text-xs mb-1 inline-block">Обработано</span>
                        <video src="/api/result/${state.jobId}/${outName}" controls muted class="w-full rounded-lg border border-gray-700"></video>
                    </div>
                </div>
                <button class="btn-secondary w-full" onclick="downloadFile('${outName}')">⬇ Скачать</button>
            `;
        } else {
            card.innerHTML = `
                <p class="text-sm font-semibold text-gray-300">${f.original_name}</p>
                <div class="compare-container" id="${compareId}">
                    <img src="/api/preview/${state.jobId}/${f.stored_name}" class="compare-before" draggable="false">
                    <div class="compare-after">
                        <img src="/api/result/${state.jobId}/${outName}" draggable="false">
                    </div>
                    <div class="compare-handle" style="left:50%"></div>
                    <span class="compare-label compare-label-before">До</span>
                    <span class="compare-label compare-label-after">После</span>
                </div>
                <button class="btn-secondary w-full" onclick="downloadFile('${outName}')">⬇ Скачать</button>
            `;
        }
        resultsGrid.appendChild(card);

        // Activate compare slider for images
        if (!isVideo) {
            requestAnimationFrame(() => initCompare(compareId));
        }
    });
}

function nameToMasked(name) {
    const dot = name.lastIndexOf(".");
    if (dot === -1) return name + "_stealthmasked";
    return name.substring(0, dot) + "_stealthmasked" + name.substring(dot);
}

// ---------------------------------------------------------------------------
// Compare slider
// ---------------------------------------------------------------------------
function initCompare(id) {
    const container = document.getElementById(id);
    if (!container) return;
    const handle = container.querySelector(".compare-handle");
    const afterWrap = container.querySelector(".compare-after");

    function setPosition(x) {
        const rect = container.getBoundingClientRect();
        let pct = ((x - rect.left) / rect.width) * 100;
        pct = Math.max(0, Math.min(100, pct));
        handle.style.left = pct + "%";
        afterWrap.style.clipPath = `inset(0 0 0 ${pct}%)`;
    }

    // Set initial
    afterWrap.style.clipPath = "inset(0 0 0 50%)";

    const onMove = (e) => {
        const x = e.touches ? e.touches[0].clientX : e.clientX;
        setPosition(x);
    };
    const onUp = () => {
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
        document.removeEventListener("touchmove", onMove);
        document.removeEventListener("touchend", onUp);
    };
    container.addEventListener("mousedown", (e) => {
        onMove(e);
        document.addEventListener("mousemove", onMove);
        document.addEventListener("mouseup", onUp);
    });
    container.addEventListener("touchstart", (e) => {
        onMove(e);
        document.addEventListener("touchmove", onMove);
        document.addEventListener("touchend", onUp);
    });
}

// ---------------------------------------------------------------------------
// Downloads
// ---------------------------------------------------------------------------
function downloadFile(name) {
    const a = document.createElement("a");
    a.href = `/api/download/${state.jobId}/${name}`;
    a.download = name;
    a.click();
}

downloadAllBtn.addEventListener("click", () => {
    if (!state.jobId) return;
    window.location.href = `/api/download-all/${state.jobId}`;
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function show(el) {
    el.classList.remove("section-hidden");
    el.classList.add("section-visible");
}
function hide(el) {
    el.classList.remove("section-visible");
    el.classList.add("section-hidden");
}

function showToast(msg, type = "info") {
    const t = document.createElement("div");
    t.className = "toast";
    const colors = { success: "#10b981", error: "#ef4444", warn: "#f59e0b", info: "#646cff" };
    t.style.borderColor = colors[type] || colors.info;
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => { t.remove(); }, 3500);
}

// New upload / reset
function resetAll() {
    state.jobId = null;
    state.files = [];
    state.processing = false;
    if (state.pollTimer) clearInterval(state.pollTimer);
    gallery.innerHTML = "";
    hide(gallerySection);
    hide(controlsSection);
    hide(progressSection);
    hide(resultsSection);
    processBtn.disabled = false;
}

const newUploadBtn = $("#new-upload-btn");
if (newUploadBtn) newUploadBtn.addEventListener("click", resetAll);
