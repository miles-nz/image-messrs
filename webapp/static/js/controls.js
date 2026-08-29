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
  const applyEffectBtn = document.getElementById("apply-effect-btn");
  const applyStatusEl = document.getElementById("apply-status");
  const compareToggleBtn = document.getElementById("compare-toggle-btn");
  const compareOverlayEl = document.getElementById("compare-overlay");
  const cropToggleBtn = document.getElementById("crop-toggle-btn");
  const cropControlsEl = document.getElementById("crop-controls");
  const cropAspectSelect = document.getElementById("crop-aspect-select");
  const cropResetBtn = document.getElementById("crop-reset-btn");
  const compareBeforeWrapEl = document.getElementById("compare-before-wrap");
  const compareBeforeImg = document.getElementById("compare-before-image");
  const compareDividerEl = document.getElementById("compare-divider");
  const zoomBtn = document.getElementById("zoom-btn");
  const zoomLightboxEl = document.getElementById("zoom-lightbox");
  const zoomViewportEl = document.getElementById("zoom-viewport");
  const zoomImageEl = document.getElementById("zoom-image");
  const zoomLoadingEl = document.getElementById("zoom-loading");
  const zoomLevelLabel = document.getElementById("zoom-level-label");
  const zoomInBtn = document.getElementById("zoom-in-btn");
  const zoomOutBtn = document.getElementById("zoom-out-btn");
  const zoomFitBtn = document.getElementById("zoom-reset-fit-btn");
  const zoomActualSizeBtn = document.getElementById("zoom-reset-100-btn");
  const zoomCloseBtn = document.getElementById("zoom-close-btn");
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
  let compareEnabled = false;
  let compareDragging = false;
  let cropEnabled = false;
  let comparePos = 50;
  const elasticCompare = compareDividerEl
    ? window.ElasticCompare.attach(compareDividerEl, compareDividerEl.querySelector(".compare-divider-handle"))
    : null;
  let zoomObjectUrl = null;
  let zoomScale = 1;
  let zoomFitScale = 1;
  let zoomPanning = false;
  let zoomPanStartX = 0;
  let zoomPanStartY = 0;
  let zoomScrollStartLeft = 0;
  let zoomScrollStartTop = 0;

  previewImg.addEventListener("load", () => {
    if (previewImg.naturalWidth && previewImg.naturalHeight) {
      previewDimsEl.textContent = `${previewImg.naturalWidth} × ${previewImg.naturalHeight}px`;
    }
    if (compareEnabled) resizeCompareOverlay();
  });

  function applyComparePos() {
    if (!compareOverlayEl) return;
    compareBeforeWrapEl.style.width = comparePos + "%";
    compareDividerEl.style.left = comparePos + "%";
  }

  function resizeCompareOverlay() {
    if (!compareOverlayEl) return;
    if (elasticCompare) elasticCompare.release();
    const rect = previewImg.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    compareOverlayEl.style.left = previewImg.offsetLeft + "px";
    compareOverlayEl.style.top = previewImg.offsetTop + "px";
    compareOverlayEl.style.width = rect.width + "px";
    compareOverlayEl.style.height = rect.height + "px";
    compareBeforeImg.style.width = rect.width + "px";
    compareBeforeImg.style.height = rect.height + "px";
    applyComparePos();
  }

  function setComparePosFromClientX(clientX) {
    const rect = compareOverlayEl.getBoundingClientRect();
    if (!rect.width) return;
    const rawPercent = ((clientX - rect.left) / rect.width) * 100;
    comparePos = Math.min(100, Math.max(0, rawPercent));
    applyComparePos();
    if (elasticCompare) {
      const overflowPercent = rawPercent < 0 ? rawPercent : rawPercent > 100 ? rawPercent - 100 : 0;
      elasticCompare.update((overflowPercent / 100) * rect.width);
    }
  }

  function setCompareEnabled(value) {
    compareEnabled = value;
    compareOverlayEl.classList.toggle("hidden", !value);
    compareToggleBtn.classList.toggle("active", value);
    if (value) resizeCompareOverlay();
    else if (elasticCompare) elasticCompare.release();
  }

  function refreshImageAReferences() {
    const t = Date.now();
    compareBeforeImg.src = `/images/${sessionId}/crop_preview?t=${t}`;
    if (window.MaskedHeading) window.MaskedHeading.refresh(`/images/${sessionId}/original?t=${t}`);
  }

  if (compareToggleBtn) {
    compareToggleBtn.addEventListener("click", () => setCompareEnabled(!compareEnabled));
  }

  function postCrop(rect) {
    const formData = new FormData();
    if (rect) {
      formData.append("x", rect.x);
      formData.append("y", rect.y);
      formData.append("width", rect.width);
      formData.append("height", rect.height);
    } else {
      formData.append("clear", "true");
    }
    fetch(`/images/${sessionId}/crop`, { method: "POST", body: formData })
      .then((res) => {
        if (!res.ok) throw new Error("crop update failed");
        refreshImageAReferences();
      })
      .catch((err) => console.error(err));
  }

  function setCropEnabled(value) {
    cropEnabled = value;
    cropToggleBtn.classList.toggle("active", value);
    cropControlsEl.classList.toggle("hidden", !value);
    if (window.CropEditor) window.CropEditor.setEnabled(value);
    if (value) {
      previewImg.src = `/images/${sessionId}/original?t=${Date.now()}`;
    } else {
      runPreview();
    }
  }

  if (cropToggleBtn) {
    cropToggleBtn.addEventListener("click", () => setCropEnabled(!cropEnabled));
  }
  if (cropResetBtn) {
    cropResetBtn.addEventListener("click", () => {
      if (window.CropEditor) window.CropEditor.reset();
    });
  }
  if (cropAspectSelect) {
    cropAspectSelect.addEventListener("change", () => {
      if (window.CropEditor) window.CropEditor.setAspect(cropAspectSelect.value);
    });
  }
  window.addEventListener("crop-updated", (e) => postCrop(e.detail));

  if (compareOverlayEl) {
    compareOverlayEl.addEventListener("mousedown", (e) => {
      compareDragging = true;
      setComparePosFromClientX(e.clientX);
      e.preventDefault();
    });
    compareOverlayEl.addEventListener(
      "touchstart",
      (e) => {
        compareDragging = true;
        setComparePosFromClientX(e.touches[0].clientX);
      },
      { passive: true }
    );
    window.addEventListener("mousemove", (e) => {
      if (!compareDragging) return;
      setComparePosFromClientX(e.clientX);
    });
    window.addEventListener(
      "touchmove",
      (e) => {
        if (!compareDragging) return;
        setComparePosFromClientX(e.touches[0].clientX);
      },
      { passive: true }
    );
    window.addEventListener("mouseup", () => {
      if (compareDragging && elasticCompare) elasticCompare.release();
      compareDragging = false;
    });
    window.addEventListener("touchend", () => {
      if (compareDragging && elasticCompare) elasticCompare.release();
      compareDragging = false;
    });
  }

  window.addEventListener("resize", () => {
    if (compareEnabled) resizeCompareOverlay();
  });

  function applyZoomScale() {
    if (!zoomImageEl.naturalWidth) return;
    const height = Math.round(zoomImageEl.naturalHeight * zoomScale);
    zoomImageEl.style.width = Math.round(zoomImageEl.naturalWidth * zoomScale) + "px";
    zoomImageEl.style.height = height + "px";
    // text-align:center handles horizontal centering (and stays fully scrollable
    // when the image overflows); vertical centering needs a margin since there's
    // no vertical equivalent that keeps the same scroll-safe behavior.
    const vpHeight = zoomViewportEl.clientHeight;
    zoomImageEl.style.marginTop = height < vpHeight ? Math.round((vpHeight - height) / 2) + "px" : "0";
    zoomLevelLabel.textContent = Math.round(zoomScale * 100) + "%";
  }

  function setZoomScale(scale) {
    zoomScale = Math.min(8, Math.max(0.05, scale));
    applyZoomScale();
  }

  function centerZoomScroll() {
    zoomViewportEl.scrollLeft = Math.max(0, (zoomImageEl.offsetWidth - zoomViewportEl.clientWidth) / 2);
    zoomViewportEl.scrollTop = Math.max(0, (zoomImageEl.offsetHeight - zoomViewportEl.clientHeight) / 2);
  }

  function openZoom() {
    const effect = effectsByName[effectSelect.value];
    if (effect.multi_image && !hasImageB) return;

    zoomLightboxEl.classList.remove("hidden");
    zoomLoadingEl.classList.remove("hidden");
    zoomLoadingEl.textContent = "Loading full-resolution image…";
    zoomImageEl.classList.add("hidden");

    const formData = buildFormData(effect);
    fetch(`/images/${sessionId}/render`, { method: "POST", body: formData })
      .then((res) => {
        if (!res.ok) throw new Error("render failed");
        return res.blob();
      })
      .then((blob) => {
        if (zoomObjectUrl) URL.revokeObjectURL(zoomObjectUrl);
        zoomObjectUrl = URL.createObjectURL(blob);
        zoomImageEl.src = zoomObjectUrl;
      })
      .catch((err) => {
        console.error(err);
        zoomLoadingEl.textContent = "Failed to load full-resolution image.";
      });
  }

  function closeZoom() {
    zoomLightboxEl.classList.add("hidden");
    if (zoomObjectUrl) {
      URL.revokeObjectURL(zoomObjectUrl);
      zoomObjectUrl = null;
    }
    zoomImageEl.removeAttribute("src");
  }

  zoomImageEl.addEventListener("load", () => {
    if (!zoomImageEl.naturalWidth || !zoomImageEl.naturalHeight) return;
    zoomLoadingEl.classList.add("hidden");
    zoomImageEl.classList.remove("hidden");
    const vpRect = zoomViewportEl.getBoundingClientRect();
    zoomFitScale = Math.min(
      1,
      (vpRect.width - 32) / zoomImageEl.naturalWidth,
      (vpRect.height - 32) / zoomImageEl.naturalHeight
    );
    if (!(zoomFitScale > 0)) zoomFitScale = 1;
    zoomScale = zoomFitScale;
    applyZoomScale();
    centerZoomScroll();
  });

  if (zoomBtn) zoomBtn.addEventListener("click", openZoom);
  if (zoomCloseBtn) zoomCloseBtn.addEventListener("click", closeZoom);
  if (zoomInBtn) zoomInBtn.addEventListener("click", () => setZoomScale(zoomScale * 1.4));
  if (zoomOutBtn) zoomOutBtn.addEventListener("click", () => setZoomScale(zoomScale / 1.4));
  if (zoomFitBtn) {
    zoomFitBtn.addEventListener("click", () => {
      setZoomScale(zoomFitScale);
      centerZoomScroll();
    });
  }
  if (zoomActualSizeBtn) {
    zoomActualSizeBtn.addEventListener("click", () => {
      setZoomScale(1);
      centerZoomScroll();
    });
  }

  if (zoomViewportEl) {
    zoomViewportEl.addEventListener(
      "wheel",
      (e) => {
        if (zoomLightboxEl.classList.contains("hidden")) return;
        e.preventDefault();
        setZoomScale(zoomScale * (e.deltaY < 0 ? 1.1 : 0.9));
      },
      { passive: false }
    );
    zoomViewportEl.addEventListener("mousedown", (e) => {
      zoomPanning = true;
      zoomPanStartX = e.clientX;
      zoomPanStartY = e.clientY;
      zoomScrollStartLeft = zoomViewportEl.scrollLeft;
      zoomScrollStartTop = zoomViewportEl.scrollTop;
      zoomViewportEl.classList.add("dragging");
      e.preventDefault();
    });
    window.addEventListener("mousemove", (e) => {
      if (!zoomPanning) return;
      zoomViewportEl.scrollLeft = zoomScrollStartLeft - (e.clientX - zoomPanStartX);
      zoomViewportEl.scrollTop = zoomScrollStartTop - (e.clientY - zoomPanStartY);
    });
    window.addEventListener("mouseup", () => {
      zoomPanning = false;
      zoomViewportEl.classList.remove("dragging");
    });
  }

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && zoomLightboxEl && !zoomLightboxEl.classList.contains("hidden")) closeZoom();
  });

  const CATEGORY_ORDER = ["glitch", "blend", "color", "distort", "video", "seam_carve"];
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
      } else if (param.kind === "color") {
        const picker = document.createElement("div");
        picker.className = "color-picker";

        input = document.createElement("input");
        input.type = "color";
        input.value = param.default || "#ff8c3c";
        if (param.description) input.title = param.description;

        const swatchRow = document.createElement("div");
        swatchRow.className = "color-swatch-row";
        const syncActiveSwatch = () => {
          swatchRow.querySelectorAll(".color-swatch").forEach((btn) => {
            btn.classList.toggle("active", btn.dataset.hex.toLowerCase() === input.value.toLowerCase());
          });
        };
        (param.choices || []).forEach((hex) => {
          const swatchBtn = document.createElement("button");
          swatchBtn.type = "button";
          swatchBtn.className = "color-swatch";
          swatchBtn.style.background = hex;
          swatchBtn.title = hex;
          swatchBtn.dataset.hex = hex;
          if (hex.toLowerCase() === input.value.toLowerCase()) swatchBtn.classList.add("active");
          swatchBtn.addEventListener("click", () => {
            input.value = hex;
            syncActiveSwatch();
            schedulePreview();
          });
          swatchRow.appendChild(swatchBtn);
        });

        input.addEventListener("input", () => {
          syncActiveSwatch();
          schedulePreview();
        });

        picker.appendChild(swatchRow);
        picker.appendChild(input);
        wrapper.appendChild(picker);
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
    if (applyEffectBtn) applyEffectBtn.disabled = needsSecondImage;
    if (zoomBtn) zoomBtn.disabled = needsSecondImage;
    thumbBoxB.classList.toggle("hidden", !(effect.multi_image && hasImageB));

    const canAnimate = !needsSecondImage && animatableParams(effect).length > 0;
    animatePanelEl.classList.toggle("hidden", !canAnimate);
    if (canAnimate) populateAnimateParamSelect(effect);

    if (compareToggleBtn) {
      const compareAllowed = !effect.accepts_mask;
      compareToggleBtn.classList.toggle("hidden", !compareAllowed);
      if (!compareAllowed) setCompareEnabled(false);
    }
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
        if (window.CropEditor) window.CropEditor.reset();
        refreshImageAReferences();
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
        if (slot === "a") {
          if (window.CropEditor) window.CropEditor.reset();
          refreshImageAReferences();
        }
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

  function applyEffect() {
    const effect = effectsByName[effectSelect.value];
    if (effect.multi_image && !hasImageB) return;

    if (window.showConfirmModal) {
      window.showConfirmModal({
        title: "Apply effect?",
        message:
          "This bakes the current effect into the photo so you can stack another one on top. You won't be able to re-tune this effect's settings afterward.",
        confirmLabel: "Apply",
        onConfirm: doApplyEffect,
      });
    } else {
      doApplyEffect();
    }
  }

  function doApplyEffect() {
    const effect = effectsByName[effectSelect.value];
    applyEffectBtn.disabled = true;
    downloadBtn.disabled = true;
    applyStatusEl.classList.remove("error", "hidden");
    applyStatusEl.textContent = "Applying effect…";

    const formData = buildFormData(effect);
    fetch(`/images/${sessionId}/apply`, { method: "POST", body: formData })
      .then((res) => {
        if (!res.ok) return res.text().then((text) => Promise.reject(new Error(text)));
        return res.json();
      })
      .then(() => {
        if (window.MaskEditor) window.MaskEditor.clear();
        if (window.CropEditor) window.CropEditor.reset();
        setCompareEnabled(false);
        thumbAImg.src = `/images/${sessionId}/thumbnail/a?t=${Date.now()}`;
        refreshImageAReferences();
        buildControlsForEffect(effect); // reset sliders to defaults for the next pass
        applyStatusEl.textContent = "Applied - pick another effect (or the same one again) to keep going.";
        runPreview();
      })
      .catch((err) => {
        console.error(err);
        applyStatusEl.textContent = "Failed to apply effect: " + err.message;
        applyStatusEl.classList.add("error");
      })
      .finally(() => {
        updateVisibility(effectsByName[effectSelect.value]);
      });
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
  if (applyEffectBtn) applyEffectBtn.addEventListener("click", applyEffect);
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
    window.confirmBeforeNav(document.getElementById("nav-image-link"), abandonMessage);
    window.confirmBeforeNav(document.getElementById("nav-video-link"), abandonMessage);
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
