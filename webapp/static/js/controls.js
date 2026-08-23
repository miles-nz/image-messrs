(function () {
  const effectsData = JSON.parse(document.getElementById("effects-data").textContent);
  const effectsByName = Object.fromEntries(effectsData.map((e) => [e.name, e]));

  const editorEl = document.querySelector(".editor");
  const sessionId = editorEl.dataset.sessionId;
  let hasImageB = editorEl.dataset.hasImageB === "true";
  const suggestedSourceCamera = editorEl.dataset.suggestedSourceCamera || "";

  const modeSelect = document.getElementById("mode-select");
  const effectSelect = document.getElementById("effect-select");
  const paramControlsEl = document.getElementById("param-controls");
  const effectAboutEl = document.getElementById("effect-about");
  const previewImg = document.getElementById("preview-image");
  const previewLoadingEl = document.getElementById("preview-loading");
  const previewDimsEl = document.getElementById("preview-dims");
  const previewStatusEl = document.getElementById("preview-status");
  const maskControlsEl = document.getElementById("mask-controls");
  const secondImagePromptEl = document.getElementById("second-image-prompt");
  const secondImageFormEl = document.getElementById("second-image-form");
  const secondImageInputEl = document.getElementById("second-image-input");
  const downloadBtn = document.getElementById("download-btn");
  const thumbBoxA = document.getElementById("thumb-box-a");
  const thumbBoxB = document.getElementById("thumb-box-b");
  const thumbAImg = document.getElementById("thumb-a-img");
  const thumbBImg = document.getElementById("thumb-b-img");
  const thumbAInput = document.getElementById("thumb-a-input");
  const thumbBInput = document.getElementById("thumb-b-input");
  const swapPhotosBtn = document.getElementById("swap-photos-btn");
  const animatePanelEl = document.getElementById("animate-panel");
  const animateParamSelect = document.getElementById("animate-param-select");
  const animateStartInput = document.getElementById("animate-start");
  const animateEndInput = document.getElementById("animate-end");
  const animateDurationInput = document.getElementById("animate-duration");
  const animateFpsInput = document.getElementById("animate-fps");
  const animateLoopStyleSelect = document.getElementById("animate-loop-style");
  const animateSeedRowEl = document.getElementById("animate-seed-row");
  const animateVarySeedCheckbox = document.getElementById("animate-vary-seed");
  const animateFullResCheckbox = document.getElementById("animate-full-res");
  const animateGenerateBtn = document.getElementById("animate-generate-btn");
  const animateStatusEl = document.getElementById("animate-status");
  const animateResultVideo = document.getElementById("animate-result-video");
  const animateDownloadLink = document.getElementById("animate-download-link");

  let currentObjectUrl = null;
  let debounceTimer = null;
  let previewRequestSeq = 0;
  let currentPreviewController = null;
  let animatePollTimer = null;

  previewImg.addEventListener("load", () => {
    if (previewImg.naturalWidth && previewImg.naturalHeight) {
      previewDimsEl.textContent = `${previewImg.naturalWidth} × ${previewImg.naturalHeight}px`;
    }
  });

  const CATEGORY_ORDER = ["seam_carve", "glitch", "blend", "color", "distort", "video"];
  const CATEGORY_LABELS = {
    seam_carve: "Seam Carve",
    glitch: "Glitch",
    blend: "Blend",
    color: "Color",
    distort: "Distort",
    video: "Video",
  };

  function effectsForMode(mode) {
    return effectsData.filter((e) => e.multi_image === (mode === "multi"));
  }

  function populateEffectSelect(mode) {
    effectSelect.innerHTML = "";
    const effects = effectsForMode(mode);
    const categories = CATEGORY_ORDER.filter((cat) => effects.some((e) => e.category === cat));
    categories.forEach((category) => {
      const group = document.createElement("optgroup");
      group.label = CATEGORY_LABELS[category] || category;
      effects
        .filter((e) => e.category === category)
        .forEach((effect) => {
          const opt = document.createElement("option");
          opt.value = effect.name;
          opt.textContent = effect.label;
          opt.title = effect.description || "";
          group.appendChild(opt);
        });
      effectSelect.appendChild(group);
    });
  }

  function buildControlsForEffect(effect) {
    paramControlsEl.innerHTML = "";

    effectSelect.title = effect.description || "";
    Array.from(effectSelect.options).forEach((opt) => {
      const optEffect = effectsByName[opt.value];
      if (optEffect) opt.title = optEffect.description || "";
    });

    effect.params.forEach((param) => {
      if (param.kind === "mask") return; // handled by the mask canvas

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
          schedulePreview();
        });
        wrapper.appendChild(input);
        wrapper.appendChild(valueLabel);
      } else if (param.kind === "bool") {
        input = document.createElement("input");
        input.type = "checkbox";
        input.checked = !!param.default;
        if (param.description) input.title = param.description;
        input.addEventListener("change", schedulePreview);
        wrapper.appendChild(input);
      } else if (param.kind === "choice") {
        input = document.createElement("select");
        if (param.description) input.title = param.description;
        const preselect =
          effect.name === "vintage_camera_profile" &&
          param.name === "source_camera" &&
          (param.choices || []).includes(suggestedSourceCamera)
            ? suggestedSourceCamera
            : param.default;
        (param.choices || []).forEach((choice) => {
          const opt = document.createElement("option");
          opt.value = choice;
          opt.textContent = choice;
          if (choice === preselect) opt.selected = true;
          input.appendChild(opt);
        });
        input.addEventListener("change", schedulePreview);
        wrapper.appendChild(input);
      } else {
        return;
      }

      input.dataset.paramName = param.name;
      paramControlsEl.appendChild(wrapper);
    });
  }

  function updateEffectAbout(effect) {
    if (!effectAboutEl) return;
    const about = effect.about || {};
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

  function buildFormData(effect) {
    const formData = new FormData();
    formData.append("effect", effect.name);
    currentParamInputs().forEach((input) => {
      const name = input.dataset.paramName;
      formData.append(name, input.type === "checkbox" ? (input.checked ? "true" : "false") : input.value);
    });
    if (effect.accepts_mask && window.MaskEditor) {
      const maskDataUrl = window.MaskEditor.getDataURL();
      if (maskDataUrl) formData.append("mask", maskDataUrl);
    }
    return formData;
  }

  function updateVisibility(effect) {
    const needsSecondImage = effect.multi_image && !hasImageB;
    secondImagePromptEl.classList.toggle("hidden", !needsSecondImage);
    swapPhotosBtn.classList.toggle("hidden", !hasImageB);
    maskControlsEl.classList.toggle("hidden", !effect.accepts_mask);
    if (window.MaskEditor) window.MaskEditor.setEnabled(effect.accepts_mask);
    downloadBtn.disabled = needsSecondImage;
    thumbBoxB.classList.toggle("hidden", !(effect.multi_image && hasImageB));

    const canAnimate = !needsSecondImage && animatableParams(effect).length > 0;
    animatePanelEl.classList.toggle("hidden", !canAnimate);
    if (canAnimate) populateAnimateParamSelect(effect);
  }

  function animatableParams(effect) {
    return effect.params.filter((p) => (p.kind === "float" || p.kind === "int") && p.min != null && p.max != null);
  }

  function seedParamFor(effect) {
    return effect.params.find((p) => p.name === "seed" && p.kind === "int") || null;
  }

  function populateAnimateParamSelect(effect) {
    animateParamSelect.innerHTML = "";
    animatableParams(effect).forEach((param) => {
      const opt = document.createElement("option");
      opt.value = param.name;
      opt.textContent = param.label;
      opt.title = param.description || "";
      animateParamSelect.appendChild(opt);
    });
    updateAnimateRangeDefaults(effect);
    updateAnimateSeedRow(effect);
  }

  function updateAnimateRangeDefaults(effect) {
    const param = animatableParams(effect).find((p) => p.name === animateParamSelect.value);
    if (!param) return;
    animateStartInput.value = param.min;
    animateEndInput.value = param.max;
  }

  function updateAnimateSeedRow(effect) {
    const seedParam = seedParamFor(effect);
    const showSeedRow = !!seedParam && seedParam.name !== animateParamSelect.value;
    animateSeedRowEl.classList.toggle("hidden", !showSeedRow);
    if (!showSeedRow) animateVarySeedCheckbox.checked = false;
  }

  function swapPhotos() {
    swapPhotosBtn.disabled = true;
    fetch(`/images/${sessionId}/swap`, { method: "POST" })
      .then((res) => {
        if (!res.ok) throw new Error("swap failed");
        return res.json();
      })
      .then(() => {
        const t = Date.now();
        thumbAImg.src = `/images/${sessionId}/thumbnail/a?t=${t}`;
        thumbBImg.src = `/images/${sessionId}/thumbnail/b?t=${t}`;
        runPreview();
      })
      .catch((err) => {
        console.error(err);
        previewStatusEl.textContent = "Failed to swap photos.";
        previewStatusEl.classList.add("error");
      })
      .finally(() => {
        swapPhotosBtn.disabled = false;
      });
  }

  function replaceImage(slot, file) {
    const box = slot === "a" ? thumbBoxA : thumbBoxB;
    const img = slot === "a" ? thumbAImg : thumbBImg;
    box.classList.add("uploading");

    const formData = new FormData();
    formData.append("image", file);
    fetch(`/images/${sessionId}/replace/${slot}`, { method: "POST", body: formData })
      .then((res) => {
        if (!res.ok) throw new Error("replace failed");
        return res.json();
      })
      .then(() => {
        img.src = `/images/${sessionId}/thumbnail/${slot}?t=${Date.now()}`;
        if (slot === "b") {
          hasImageB = true;
          updateVisibility(effectsByName[effectSelect.value]);
        }
        runPreview();
      })
      .catch((err) => {
        console.error(err);
        previewStatusEl.textContent = `Failed to replace photo ${slot.toUpperCase()}.`;
        previewStatusEl.classList.add("error");
      })
      .finally(() => box.classList.remove("uploading"));
  }

  function schedulePreview() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(runPreview, 250);
  }

  function runPreview() {
    const effect = effectsByName[effectSelect.value];
    if (effect.multi_image && !hasImageB) return;

    // Cancel any in-flight preview so a slow, superseded request can't
    // resolve after (and overwrite the result of) a newer one.
    if (currentPreviewController) currentPreviewController.abort();
    const controller = new AbortController();
    currentPreviewController = controller;
    const seq = ++previewRequestSeq;

    previewStatusEl.textContent = "";
    previewStatusEl.classList.remove("error");
    previewLoadingEl.classList.remove("hidden");

    const formData = buildFormData(effect);
    fetch(`/images/${sessionId}/preview`, { method: "POST", body: formData, signal: controller.signal })
      .then((res) => {
        if (!res.ok) throw new Error("preview failed");
        return res.blob();
      })
      .then((blob) => {
        if (seq !== previewRequestSeq) return; // a newer request already superseded this one
        const url = URL.createObjectURL(blob);
        const previous = currentObjectUrl;
        currentObjectUrl = url;
        previewImg.src = url;
        if (previous) URL.revokeObjectURL(previous);
      })
      .catch((err) => {
        if (err.name === "AbortError" || seq !== previewRequestSeq) return;
        console.error(err);
        previewStatusEl.textContent = "Preview failed with these settings.";
        previewStatusEl.classList.add("error");
      })
      .finally(() => {
        if (seq === previewRequestSeq) previewLoadingEl.classList.add("hidden");
      });
  }

  function downloadFullRes() {
    const effect = effectsByName[effectSelect.value];
    if (effect.multi_image && !hasImageB) return;
    const formData = buildFormData(effect);
    fetch(`/images/${sessionId}/render`, { method: "POST", body: formData })
      .then((res) => {
        if (!res.ok) throw new Error("render failed");
        return res.blob();
      })
      .then((blob) => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "image-messrs-output.png";
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      })
      .catch((err) => console.error(err));
  }

  function startAnimateJob() {
    const effect = effectsByName[effectSelect.value];
    const formData = buildFormData(effect);
    formData.append("sweep_param", animateParamSelect.value);
    formData.append("sweep_start", animateStartInput.value);
    formData.append("sweep_end", animateEndInput.value);
    formData.append("duration", animateDurationInput.value);
    formData.append("fps", animateFpsInput.value);
    formData.append("loop_style", animateLoopStyleSelect.value);
    formData.append("full_res", animateFullResCheckbox.checked ? "true" : "false");
    if (!animateSeedRowEl.classList.contains("hidden") && animateVarySeedCheckbox.checked) {
      const seedParam = seedParamFor(effect);
      if (seedParam) formData.append("seed_param", seedParam.name);
    }

    animateGenerateBtn.disabled = true;
    animateStatusEl.textContent = "Starting…";
    animateStatusEl.classList.remove("error");
    animateResultVideo.classList.add("hidden");
    animateDownloadLink.classList.add("hidden");

    fetch(`/images/${sessionId}/animate`, { method: "POST", body: formData })
      .then((res) => {
        if (!res.ok) return res.text().then((text) => Promise.reject(new Error(text)));
        return res.json();
      })
      .then(({ job_id }) => pollAnimateJob(job_id))
      .catch((err) => {
        animateStatusEl.textContent = "Failed to start: " + err.message;
        animateStatusEl.classList.add("error");
        animateGenerateBtn.disabled = false;
      });
  }

  function pollAnimateJob(jobId) {
    clearInterval(animatePollTimer);
    animatePollTimer = setInterval(() => {
      fetch(`/images/${sessionId}/animate/jobs/${jobId}/status`)
        .then((res) => res.json())
        .then((data) => {
          if (data.status === "running" || data.status === "pending") {
            const frame = data.progress && data.progress.frame;
            animateStatusEl.textContent = frame ? `Rendering… frame ${frame}` : "Rendering…";
          } else if (data.status === "done") {
            clearInterval(animatePollTimer);
            animateStatusEl.textContent = "Done.";
            animateGenerateBtn.disabled = false;
            const url = `/images/${sessionId}/animate/jobs/${jobId}/result`;
            animateResultVideo.src = url;
            animateResultVideo.classList.remove("hidden");
            animateDownloadLink.href = url;
            animateDownloadLink.classList.remove("hidden");
          } else if (data.status === "error") {
            clearInterval(animatePollTimer);
            animateStatusEl.textContent = "Error: " + (data.error || "unknown error");
            animateStatusEl.classList.add("error");
            animateGenerateBtn.disabled = false;
          }
        })
        .catch(() => {
          clearInterval(animatePollTimer);
          animateStatusEl.textContent = "Lost connection to job status.";
          animateStatusEl.classList.add("error");
          animateGenerateBtn.disabled = false;
        });
    }, 1000);
  }

  function selectEffectChanged() {
    const effect = effectsByName[effectSelect.value];
    buildControlsForEffect(effect);
    updateEffectAbout(effect);
    updateVisibility(effect);
    runPreview();
  }

  modeSelect.addEventListener("change", () => {
    populateEffectSelect(modeSelect.value);
    selectEffectChanged();
  });
  effectSelect.addEventListener("change", selectEffectChanged);
  downloadBtn.addEventListener("click", downloadFullRes);
  swapPhotosBtn.addEventListener("click", swapPhotos);
  animateParamSelect.addEventListener("change", () => {
    const effect = effectsByName[effectSelect.value];
    updateAnimateRangeDefaults(effect);
    updateAnimateSeedRow(effect);
  });
  animateGenerateBtn.addEventListener("click", startAnimateJob);
  window.addEventListener("mask-updated", schedulePreview);
  thumbAInput.addEventListener("change", () => {
    const file = thumbAInput.files[0];
    if (file) replaceImage("a", file);
    thumbAInput.value = "";
  });
  thumbBInput.addEventListener("change", () => {
    const file = thumbBInput.files[0];
    if (file) replaceImage("b", file);
    thumbBInput.value = "";
  });
  secondImageFormEl.addEventListener("submit", (e) => {
    e.preventDefault();
    const file = secondImageInputEl.files[0];
    if (file) replaceImage("b", file);
    secondImageInputEl.value = "";
  });

  if (window.confirmBeforeNav) {
    const abandonMessage = "You'll lose this editing session and any effect settings you've configured. Continue?";
    window.confirmBeforeNav(document.getElementById("home-link"), abandonMessage);
    window.confirmBeforeNav(document.getElementById("new-upload-link"), abandonMessage);
  }

  // If both photos were already uploaded up front, default to browsing
  // blend effects rather than making the user flip the selector themselves.
  modeSelect.value = hasImageB ? "multi" : "single";
  populateEffectSelect(modeSelect.value);

  const initialEffect = effectsByName[effectSelect.value];
  buildControlsForEffect(initialEffect);
  updateEffectAbout(initialEffect);
  updateVisibility(initialEffect);
})();
