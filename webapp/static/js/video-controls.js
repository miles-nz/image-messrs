(function () {
  const techniques = JSON.parse(document.getElementById("techniques-data").textContent);
  const frameEffectsData = JSON.parse(document.getElementById("frame-effects-data").textContent);
  const frameEffectsByName = Object.fromEntries(frameEffectsData.map((e) => [e.name, e]));

  const editorEl = document.querySelector(".video-editor");
  const sessionId = editorEl.dataset.sessionId;
  const hasMotionClip = editorEl.dataset.hasMotionClip === "true";
  let thumbnailUrl = document.getElementById("thumbnail").src;

  const techniqueSelect = document.getElementById("technique-select");
  const frameEffectSelectLabel = document.getElementById("frame-effect-select-label");
  const frameEffectSelect = document.getElementById("frame-effect-select");
  const subTechniqueSelectLabel = document.getElementById("sub-technique-select-label");
  const subTechniqueSelect = document.getElementById("sub-technique-select");
  const paramControlsEl = document.getElementById("param-controls");
  const effectAboutEl = document.getElementById("effect-about");
  const motionPromptEl = document.getElementById("motion-clip-prompt");
  const previewBtn = document.getElementById("preview-btn");
  const renderBtn = document.getElementById("render-btn");
  const jobStatusEl = document.getElementById("job-status");
  const thumbnail = document.getElementById("thumbnail");
  const resultVideo = document.getElementById("result-video");
  const downloadLink = document.getElementById("download-link");
  const applyVideoBtn = document.getElementById("apply-video-btn");
  const applyStatusEl = document.getElementById("apply-status");
  const metaDurationEl = document.getElementById("meta-duration");
  const metaResolutionEl = document.getElementById("meta-resolution");
  const metaCodecEl = document.getElementById("meta-codec");
  const framePreviewLoadingEl = document.getElementById("frame-preview-loading");
  const framePreviewStatusEl = document.getElementById("frame-preview-status");
  const varyPanelEl = document.getElementById("vary-panel");
  const varyEnableCheckbox = document.getElementById("vary-enable");
  const varyControlsEl = document.getElementById("vary-controls");
  const varyParamSelect = document.getElementById("vary-param-select");
  const varyStartInput = document.getElementById("vary-start");
  const varyEndInput = document.getElementById("vary-end");
  const varyLoopStyleSelect = document.getElementById("vary-loop-style");

  const CATEGORY_ORDER = ["glitch", "blend", "color", "distort", "video", "seam_carve"];
  const CATEGORY_LABELS = {
    seam_carve: "Seam Carve",
    glitch: "Glitch",
    blend: "Blend",
    color: "Color",
    distort: "Distort",
    video: "Video",
  };

  let pollTimer = null;
  let framePreviewObjectUrl = null;
  let frameDebounceTimer = null;
  let frameRequestSeq = 0;
  let currentFramePreviewController = null;
  let lastFullRenderJobId = null;

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
    const categories = CATEGORY_ORDER.filter((cat) => frameEffectsData.some((e) => e.category === cat));
    categories.forEach((category) => {
      const group = document.createElement("optgroup");
      group.label = CATEGORY_LABELS[category] || category;
      frameEffectsData
        .filter((e) => e.category === category)
        .forEach((effect) => {
          const opt = document.createElement("option");
          opt.value = effect.name;
          opt.textContent = effect.label;
          opt.title = effect.description || "";
          group.appendChild(opt);
        });
      frameEffectSelect.appendChild(group);
    });
  }

  function animatableParams(effect) {
    return (effect ? effect.params : []).filter(
      (p) => (p.kind === "float" || p.kind === "int") && p.min != null && p.max != null
    );
  }

  function populateVaryParamSelect(effect) {
    varyParamSelect.innerHTML = "";
    animatableParams(effect).forEach((param) => {
      const opt = document.createElement("option");
      opt.value = param.name;
      opt.textContent = param.label;
      opt.title = param.description || "";
      varyParamSelect.appendChild(opt);
    });
    updateVaryRangeDefaults(effect);
  }

  function updateVaryRangeDefaults(effect) {
    const param = animatableParams(effect).find((p) => p.name === varyParamSelect.value);
    if (!param) return;
    varyStartInput.value = param.min;
    varyEndInput.value = param.max;
  }

  function updateVaryPanelVisibility() {
    const spec = currentSpec();
    const effect = isFrameBridge(spec) ? currentFrameEffect() : null;
    const canVary = !!effect && animatableParams(effect).length > 0;
    varyPanelEl.classList.toggle("hidden", !canVary);
    if (canVary) {
      populateVaryParamSelect(effect);
    } else {
      varyEnableCheckbox.checked = false;
      varyControlsEl.classList.add("hidden");
    }
  }

  function isSubTechniqueBridge(spec) {
    return !!spec.sub_techniques;
  }

  function currentSubTechnique() {
    const spec = currentSpec();
    return isSubTechniqueBridge(spec) ? spec.sub_techniques[subTechniqueSelect.value] : null;
  }

  function populateSubTechniqueSelect(spec) {
    subTechniqueSelect.innerHTML = "";
    if (!isSubTechniqueBridge(spec)) return;
    Object.entries(spec.sub_techniques).forEach(([name, subSpec]) => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = subSpec.label;
      subTechniqueSelect.appendChild(opt);
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
    updateVaryPanelVisibility();
    if (isFrameBridge(spec)) {
      const effect = currentFrameEffect();
      buildParamControls(effect ? effect.params : [], scheduleFramePreview);
      updateAbout(effect ? effect.about : spec.about);
      runFramePreview();
    } else if (isSubTechniqueBridge(spec)) {
      const subTechnique = currentSubTechnique();
      buildParamControls(subTechnique ? subTechnique.params : [], () => {});
      updateAbout((subTechnique && subTechnique.about) || spec.about);
      resetToStaticThumbnail();
    } else {
      buildParamControls(spec.params, () => {});
      updateAbout(spec.about);
      resetToStaticThumbnail();
    }
  }

  function updateVisibility(spec) {
    frameEffectSelectLabel.classList.toggle("hidden", !isFrameBridge(spec));
    subTechniqueSelectLabel.classList.toggle("hidden", !isSubTechniqueBridge(spec));
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
      if (!varyPanelEl.classList.contains("hidden") && varyEnableCheckbox.checked) {
        formData.append("vary_param", varyParamSelect.value);
        formData.append("vary_start", varyStartInput.value);
        formData.append("vary_end", varyEndInput.value);
        formData.append("vary_loop_style", varyLoopStyleSelect.value);
      }
    }
    if (isSubTechniqueBridge(spec)) {
      formData.append("sub_technique", subTechniqueSelect.value);
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
    applyVideoBtn.classList.add("hidden");
    applyStatusEl.classList.add("hidden");
    lastFullRenderJobId = null;
    thumbnail.classList.remove("hidden");

    fetch(`/video/${sessionId}/process`, { method: "POST", body: formData })
      .then((res) => {
        if (!res.ok) return res.text().then((text) => Promise.reject(new Error(text)));
        return res.json();
      })
      .then(({ job_id }) => pollJob(job_id, trim))
      .catch((err) => {
        jobStatusEl.textContent = "Failed to start: " + err.message;
        jobStatusEl.classList.add("error");
        const needsMotion = spec.needs_motion_clip && !hasMotionClip;
        previewBtn.disabled = needsMotion;
        renderBtn.disabled = needsMotion;
      });
  }

  function pollJob(jobId, trim) {
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
            if (!trim) {
              lastFullRenderJobId = jobId;
              applyVideoBtn.classList.remove("hidden");
              applyVideoBtn.disabled = false;
            }
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

  function updateMeta(meta) {
    meta = meta || {};
    metaDurationEl.textContent = meta.duration ? `${meta.duration}s` : "-";
    metaResolutionEl.textContent = `${meta.width || "?"}x${meta.height || "?"}`;
    metaCodecEl.textContent = meta.codec || "-";
  }

  function applyVideo() {
    if (!lastFullRenderJobId) return;
    if (window.showConfirmModal) {
      window.showConfirmModal({
        title: "Apply this render?",
        message:
          "This makes the rendered clip the new base video, so you can stack another technique on top of it. You won't be able to get back to the previous version afterward.",
        confirmLabel: "Apply",
        onConfirm: doApplyVideo,
      });
    } else {
      doApplyVideo();
    }
  }

  function doApplyVideo() {
    const jobId = lastFullRenderJobId;
    if (!jobId) return;

    applyVideoBtn.disabled = true;
    applyStatusEl.classList.remove("error", "hidden");
    applyStatusEl.textContent = "Applying…";

    fetch(`/video/${sessionId}/apply/${jobId}`, { method: "POST" })
      .then((res) => {
        if (!res.ok) return res.text().then((text) => Promise.reject(new Error(text)));
        return res.json();
      })
      .then(({ meta }) => {
        lastFullRenderJobId = null;
        applyVideoBtn.classList.add("hidden");
        resultVideo.classList.add("hidden");
        resultVideo.removeAttribute("src");
        downloadLink.classList.add("hidden");
        updateMeta(meta);
        thumbnailUrl = `/video/${sessionId}/thumbnail?t=${Date.now()}`;
        thumbnail.classList.remove("hidden");
        resetToStaticThumbnail();
        if (window.MaskedHeading) {
          window.MaskedHeading.refresh(`/video/${sessionId}/heading_media?t=${Date.now()}`);
        }
        applyStatusEl.textContent = "Applied - pick another technique to keep going.";
      })
      .catch((err) => {
        console.error(err);
        applyStatusEl.textContent = "Failed to apply: " + err.message;
        applyStatusEl.classList.add("error");
        applyVideoBtn.disabled = false;
      });
  }

  techniqueSelect.addEventListener("change", () => {
    const spec = currentSpec();
    populateSubTechniqueSelect(spec);
    updateVisibility(spec);
    refreshControlsAndAbout();
  });
  frameEffectSelect.addEventListener("change", refreshControlsAndAbout);
  subTechniqueSelect.addEventListener("change", refreshControlsAndAbout);
  varyEnableCheckbox.addEventListener("change", () => {
    varyControlsEl.classList.toggle("hidden", !varyEnableCheckbox.checked);
  });
  varyParamSelect.addEventListener("change", () => updateVaryRangeDefaults(currentFrameEffect()));
  previewBtn.addEventListener("click", () => startJob(true));
  renderBtn.addEventListener("click", () => startJob(false));
  applyVideoBtn.addEventListener("click", applyVideo);

  if (window.confirmBeforeNav) {
    const abandonMessage = "You'll lose this editing session and any results you haven't downloaded. Continue?";
    window.confirmBeforeNav(document.getElementById("home-link"), abandonMessage);
    window.confirmBeforeNav(document.getElementById("new-upload-link"), abandonMessage);
    window.confirmBeforeNav(document.getElementById("nav-image-link"), abandonMessage);
    window.confirmBeforeNav(document.getElementById("nav-video-link"), abandonMessage);
  }

  populateFrameEffectSelect();
  populateSubTechniqueSelect(currentSpec());
  updateVisibility(currentSpec());
  refreshControlsAndAbout();
})();
