(function () {
  const techniques = JSON.parse(document.getElementById("techniques-data").textContent);
  const frameEffectsData = JSON.parse(document.getElementById("frame-effects-data").textContent);
  const frameEffectsByName = Object.fromEntries(frameEffectsData.map((e) => [e.name, e]));

  const editorEl = document.querySelector(".video-editor");
  const sessionId = editorEl.dataset.sessionId;
  const hasMotionClip = editorEl.dataset.hasMotionClip === "true";
  const thumbnailUrl = document.getElementById("thumbnail").src;

  const techniqueSelect = document.getElementById("technique-select");
  const frameEffectSelectLabel = document.getElementById("frame-effect-select-label");
  const frameEffectSelect = document.getElementById("frame-effect-select");
  const paramControlsEl = document.getElementById("param-controls");
  const effectAboutEl = document.getElementById("effect-about");
  const motionPromptEl = document.getElementById("motion-clip-prompt");
  const previewBtn = document.getElementById("preview-btn");
  const renderBtn = document.getElementById("render-btn");
  const jobStatusEl = document.getElementById("job-status");
  const thumbnail = document.getElementById("thumbnail");
  const resultVideo = document.getElementById("result-video");
  const downloadLink = document.getElementById("download-link");
  const framePreviewLoadingEl = document.getElementById("frame-preview-loading");
  const framePreviewStatusEl = document.getElementById("frame-preview-status");

  let pollTimer = null;
  let framePreviewObjectUrl = null;
  let frameDebounceTimer = null;
  let frameRequestSeq = 0;
  let currentFramePreviewController = null;

  function currentSpec() {
    return techniques[techniqueSelect.value];
  }

  function isFrameBridge(spec) {
    return !!spec.frame_effect_bridge;
  }

  function currentFrameEffect() {
    return frameEffectsByName[frameEffectSelect.value];
  }

  function populateFrameEffectSelect() {
    frameEffectSelect.innerHTML = "";
    frameEffectsData.forEach((effect) => {
      const opt = document.createElement("option");
      opt.value = effect.name;
      opt.textContent = `${effect.label} (${effect.category})`;
      opt.title = effect.description || "";
      frameEffectSelect.appendChild(opt);
    });
  }

  function buildParamControls(params, onChange) {
    paramControlsEl.innerHTML = "";

    (params || []).forEach((param) => {
      if (param.kind === "mask") return; // not applicable per-frame

      const wrapper = document.createElement("label");
      wrapper.className = "param-control";
      if (param.description) wrapper.title = param.description;

      const labelText = document.createElement("span");
      labelText.className = "param-label";
      labelText.textContent = param.label;
      wrapper.appendChild(labelText);

      let input;
      if (param.kind === "float" || param.kind === "int") {
        input = document.createElement("input");
        input.type = "range";
        input.min = param.min ?? 0;
        input.max = param.max ?? 100;
        input.step = param.step ?? (param.kind === "int" ? 1 : 0.01);
        input.value = param.default ?? param.min ?? 0;
        if (param.description) input.title = param.description;
        const valueLabel = document.createElement("span");
        valueLabel.className = "param-value";
        valueLabel.textContent = input.value;
        input.addEventListener("input", () => {
          valueLabel.textContent = input.value;
          onChange();
        });
        wrapper.appendChild(input);
        wrapper.appendChild(valueLabel);
      } else if (param.kind === "bool") {
        input = document.createElement("input");
        input.type = "checkbox";
        input.checked = !!param.default;
        if (param.description) input.title = param.description;
        input.addEventListener("change", onChange);
        wrapper.appendChild(input);
      } else if (param.kind === "choice") {
        input = document.createElement("select");
        if (param.description) input.title = param.description;
        (param.choices || []).forEach((choice) => {
          const opt = document.createElement("option");
          opt.value = choice;
          opt.textContent = choice;
          if (choice === param.default) opt.selected = true;
          input.appendChild(opt);
        });
        input.addEventListener("change", onChange);
        wrapper.appendChild(input);
      } else {
        return;
      }

      input.dataset.paramName = param.name;
      paramControlsEl.appendChild(wrapper);
    });
  }

  function updateAbout(about) {
    if (!effectAboutEl) return;
    about = about || {};
    effectAboutEl.querySelectorAll("[data-about]").forEach((el) => {
      const text = about[el.dataset.about] || "";
      el.textContent = text;
      el.parentElement.classList.toggle("hidden", !text);
    });
    effectAboutEl.classList.toggle("hidden", Object.keys(about).length === 0);
  }

  function currentParamInputs() {
    return Array.from(paramControlsEl.querySelectorAll("[data-param-name]"));
  }

  function scheduleFramePreview() {
    if (!isFrameBridge(currentSpec())) return;
    clearTimeout(frameDebounceTimer);
    frameDebounceTimer = setTimeout(runFramePreview, 200);
  }

  function runFramePreview() {
    const spec = currentSpec();
    if (!isFrameBridge(spec)) return;

    if (currentFramePreviewController) currentFramePreviewController.abort();
    const controller = new AbortController();
    currentFramePreviewController = controller;
    const seq = ++frameRequestSeq;

    framePreviewStatusEl.textContent = "";
    framePreviewStatusEl.classList.remove("error");
    framePreviewLoadingEl.classList.remove("hidden");

    const formData = new FormData();
    formData.append("frame_effect", frameEffectSelect.value);
    currentParamInputs().forEach((input) => {
      const value = input.type === "checkbox" ? (input.checked ? "true" : "false") : input.value;
      formData.append(input.dataset.paramName, value);
    });

    fetch(`/video/${sessionId}/frame_preview`, { method: "POST", body: formData, signal: controller.signal })
      .then((res) => {
        if (!res.ok) throw new Error("preview failed");
        return res.blob();
      })
      .then((blob) => {
        if (seq !== frameRequestSeq) return;
        const url = URL.createObjectURL(blob);
        const previous = framePreviewObjectUrl;
        framePreviewObjectUrl = url;
        thumbnail.src = url;
        if (previous) URL.revokeObjectURL(previous);
      })
      .catch((err) => {
        if (err.name === "AbortError" || seq !== frameRequestSeq) return;
        console.error(err);
        framePreviewStatusEl.textContent = "Preview failed with these settings.";
        framePreviewStatusEl.classList.add("error");
      })
      .finally(() => {
        if (seq === frameRequestSeq) framePreviewLoadingEl.classList.add("hidden");
      });
  }

  function resetToStaticThumbnail() {
    clearTimeout(frameDebounceTimer);
    if (currentFramePreviewController) currentFramePreviewController.abort();
    framePreviewStatusEl.textContent = "";
    framePreviewStatusEl.classList.remove("error");
    framePreviewLoadingEl.classList.add("hidden");
    if (framePreviewObjectUrl) {
      URL.revokeObjectURL(framePreviewObjectUrl);
      framePreviewObjectUrl = null;
    }
    thumbnail.src = thumbnailUrl;
  }

  function refreshControlsAndAbout() {
    const spec = currentSpec();
    if (isFrameBridge(spec)) {
      const effect = currentFrameEffect();
      buildParamControls(effect ? effect.params : [], scheduleFramePreview);
      updateAbout(effect ? effect.about : spec.about);
      runFramePreview();
    } else {
      buildParamControls(spec.params, () => {});
      updateAbout(spec.about);
      resetToStaticThumbnail();
    }
  }

  function updateVisibility(spec) {
    frameEffectSelectLabel.classList.toggle("hidden", !isFrameBridge(spec));
    const needsMotion = spec.needs_motion_clip && !hasMotionClip;
    motionPromptEl.classList.toggle("hidden", !needsMotion);
    previewBtn.disabled = needsMotion;
    renderBtn.disabled = needsMotion;
  }

  function startJob(trim) {
    const spec = currentSpec();
    const formData = new FormData();
    formData.append("technique", techniqueSelect.value);
    if (isFrameBridge(spec)) {
      formData.append("frame_effect", frameEffectSelect.value);
    }
    formData.append("trim_preview", trim ? "true" : "false");
    currentParamInputs().forEach((input) => {
      const value = input.type === "checkbox" ? (input.checked ? "true" : "false") : input.value;
      formData.append(input.dataset.paramName, value);
    });

    previewBtn.disabled = true;
    renderBtn.disabled = true;
    jobStatusEl.textContent = trim ? "Starting preview…" : "Starting full render…";
    jobStatusEl.classList.remove("error");
    resultVideo.classList.add("hidden");
    downloadLink.classList.add("hidden");
    thumbnail.classList.remove("hidden");

    fetch(`/video/${sessionId}/process`, { method: "POST", body: formData })
      .then((res) => {
        if (!res.ok) return res.text().then((text) => Promise.reject(new Error(text)));
        return res.json();
      })
      .then(({ job_id }) => pollJob(job_id))
      .catch((err) => {
        jobStatusEl.textContent = "Failed to start: " + err.message;
        jobStatusEl.classList.add("error");
        const needsMotion = spec.needs_motion_clip && !hasMotionClip;
        previewBtn.disabled = needsMotion;
        renderBtn.disabled = needsMotion;
      });
  }

  function pollJob(jobId) {
    clearInterval(pollTimer);
    pollTimer = setInterval(() => {
      fetch(`/video/${sessionId}/jobs/${jobId}/status`)
        .then((res) => res.json())
        .then((data) => {
          if (data.status === "running" || data.status === "pending") {
            const frame = data.progress && data.progress.frame;
            jobStatusEl.textContent = frame ? `Processing… frame ${frame}` : "Processing…";
          } else if (data.status === "done") {
            clearInterval(pollTimer);
            jobStatusEl.textContent = "Done.";
            previewBtn.disabled = false;
            renderBtn.disabled = false;
            const url = `/video/${sessionId}/jobs/${jobId}/result`;
            resultVideo.src = url;
            resultVideo.classList.remove("hidden");
            thumbnail.classList.add("hidden");
            downloadLink.href = url;
            downloadLink.classList.remove("hidden");
          } else if (data.status === "error") {
            clearInterval(pollTimer);
            jobStatusEl.textContent = "Error: " + (data.error || "unknown error");
            jobStatusEl.classList.add("error");
            previewBtn.disabled = false;
            renderBtn.disabled = false;
          }
        })
        .catch(() => {
          clearInterval(pollTimer);
          jobStatusEl.textContent = "Lost connection to job status.";
          jobStatusEl.classList.add("error");
          previewBtn.disabled = false;
          renderBtn.disabled = false;
        });
    }, 1000);
  }

  techniqueSelect.addEventListener("change", () => {
    const spec = currentSpec();
    updateVisibility(spec);
    refreshControlsAndAbout();
  });
  frameEffectSelect.addEventListener("change", refreshControlsAndAbout);
  previewBtn.addEventListener("click", () => startJob(true));
  renderBtn.addEventListener("click", () => startJob(false));

  if (window.confirmBeforeNav) {
    const abandonMessage = "You'll lose this editing session and any results you haven't downloaded. Continue?";
    window.confirmBeforeNav(document.getElementById("home-link"), abandonMessage);
    window.confirmBeforeNav(document.getElementById("new-upload-link"), abandonMessage);
  }

  populateFrameEffectSelect();
  updateVisibility(currentSpec());
  refreshControlsAndAbout();
})();
