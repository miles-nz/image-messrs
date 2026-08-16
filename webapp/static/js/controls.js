(function () {
  const effectsData = JSON.parse(document.getElementById("effects-data").textContent);
  const effectsByName = Object.fromEntries(effectsData.map((e) => [e.name, e]));

  const editorEl = document.querySelector(".editor");
  const sessionId = editorEl.dataset.sessionId;
  let hasImageB = editorEl.dataset.hasImageB === "true";

  const effectSelect = document.getElementById("effect-select");
  const paramControlsEl = document.getElementById("param-controls");
  const effectAboutEl = document.getElementById("effect-about");
  const previewImg = document.getElementById("preview-image");
  const previewLoadingEl = document.getElementById("preview-loading");
  const previewDimsEl = document.getElementById("preview-dims");
  const previewStatusEl = document.getElementById("preview-status");
  const maskControlsEl = document.getElementById("mask-controls");
  const secondImagePromptEl = document.getElementById("second-image-prompt");
  const swapControlEl = document.getElementById("swap-images-control");
  const swapCheckbox = document.getElementById("swap-images");
  const downloadBtn = document.getElementById("download-btn");
  const thumbBoxA = document.getElementById("thumb-box-a");
  const thumbBoxB = document.getElementById("thumb-box-b");
  const thumbAImg = document.getElementById("thumb-a-img");
  const thumbBImg = document.getElementById("thumb-b-img");
  const thumbAInput = document.getElementById("thumb-a-input");
  const thumbBInput = document.getElementById("thumb-b-input");

  let currentObjectUrl = null;
  let debounceTimer = null;
  let previewRequestSeq = 0;
  let currentPreviewController = null;

  previewImg.addEventListener("load", () => {
    if (previewImg.naturalWidth && previewImg.naturalHeight) {
      previewDimsEl.textContent = `${previewImg.naturalWidth} × ${previewImg.naturalHeight}px`;
    }
  });

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
        (param.choices || []).forEach((choice) => {
          const opt = document.createElement("option");
          opt.value = choice;
          opt.textContent = choice;
          if (choice === param.default) opt.selected = true;
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
    if (effect.multi_image) {
      formData.append("swap_images", swapCheckbox.checked ? "true" : "false");
    }
    return formData;
  }

  function updateVisibility(effect) {
    const needsSecondImage = effect.multi_image && !hasImageB;
    secondImagePromptEl.classList.toggle("hidden", !needsSecondImage);
    swapControlEl.classList.toggle("hidden", !(effect.multi_image && hasImageB));
    maskControlsEl.classList.toggle("hidden", !effect.accepts_mask);
    if (window.MaskEditor) window.MaskEditor.setEnabled(effect.accepts_mask);
    downloadBtn.disabled = needsSecondImage;
    thumbBoxB.classList.toggle("hidden", !(effect.multi_image && hasImageB));
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

  effectSelect.addEventListener("change", () => {
    const effect = effectsByName[effectSelect.value];
    buildControlsForEffect(effect);
    updateEffectAbout(effect);
    updateVisibility(effect);
    runPreview();
  });
  downloadBtn.addEventListener("click", downloadFullRes);
  swapCheckbox.addEventListener("change", schedulePreview);
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

  if (window.confirmBeforeNav) {
    const abandonMessage = "You'll lose this editing session and any effect settings you've configured. Continue?";
    window.confirmBeforeNav(document.getElementById("home-link"), abandonMessage);
    window.confirmBeforeNav(document.getElementById("new-upload-link"), abandonMessage);
  }

  const initialEffect = effectsByName[effectSelect.value];
  buildControlsForEffect(initialEffect);
  updateEffectAbout(initialEffect);
  updateVisibility(initialEffect);
})();
