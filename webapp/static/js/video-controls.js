(function () {
  const techniques = JSON.parse(document.getElementById("techniques-data").textContent);

  const editorEl = document.querySelector(".video-editor");
  const sessionId = editorEl.dataset.sessionId;
  const hasMotionClip = editorEl.dataset.hasMotionClip === "true";

  const techniqueSelect = document.getElementById("technique-select");
  const paramControlsEl = document.getElementById("param-controls");
  const effectAboutEl = document.getElementById("effect-about");
  const motionPromptEl = document.getElementById("motion-clip-prompt");
  const processBtn = document.getElementById("process-btn");
  const jobStatusEl = document.getElementById("job-status");
  const thumbnail = document.getElementById("thumbnail");
  const resultVideo = document.getElementById("result-video");
  const downloadLink = document.getElementById("download-link");
  const trimCheckbox = document.getElementById("trim-preview");

  let pollTimer = null;

  function currentSpec() {
    return techniques[techniqueSelect.value];
  }

  function buildControls(spec) {
    paramControlsEl.innerHTML = "";

    techniqueSelect.title = spec.description || "";
    Array.from(techniqueSelect.options).forEach((opt) => {
      const optSpec = techniques[opt.value];
      if (optSpec) opt.title = optSpec.description || "";
    });

    spec.params.forEach((param) => {
      const wrapper = document.createElement("label");
      wrapper.className = "param-control";
      if (param.description) wrapper.title = param.description;

      const labelText = document.createElement("span");
      labelText.className = "param-label";
      labelText.textContent = param.label;
      wrapper.appendChild(labelText);

      const input = document.createElement("input");
      input.type = "range";
      input.min = param.min;
      input.max = param.max;
      input.step = param.step;
      input.value = param.default;
      input.dataset.paramName = param.name;
      if (param.description) input.title = param.description;

      const valueLabel = document.createElement("span");
      valueLabel.className = "param-value";
      valueLabel.textContent = input.value;
      input.addEventListener("input", () => {
        valueLabel.textContent = input.value;
      });

      wrapper.appendChild(input);
      wrapper.appendChild(valueLabel);
      paramControlsEl.appendChild(wrapper);
    });
  }

  function updateAbout(spec) {
    if (!effectAboutEl) return;
    const about = spec.about || {};
    effectAboutEl.querySelectorAll("[data-about]").forEach((el) => {
      const text = about[el.dataset.about] || "";
      el.textContent = text;
      el.parentElement.classList.toggle("hidden", !text);
    });
    effectAboutEl.classList.toggle("hidden", Object.keys(about).length === 0);
  }

  function updateVisibility(spec) {
    const needsMotion = spec.needs_motion_clip && !hasMotionClip;
    motionPromptEl.classList.toggle("hidden", !needsMotion);
    processBtn.disabled = needsMotion;
  }

  function startJob() {
    const spec = currentSpec();
    const formData = new FormData();
    formData.append("technique", techniqueSelect.value);
    formData.append("trim_preview", trimCheckbox.checked ? "true" : "false");
    paramControlsEl.querySelectorAll("[data-param-name]").forEach((input) => {
      formData.append(input.dataset.paramName, input.value);
    });

    processBtn.disabled = true;
    jobStatusEl.textContent = "Starting…";
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
        processBtn.disabled = spec.needs_motion_clip && !hasMotionClip;
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
            processBtn.disabled = false;
            const url = `/video/${sessionId}/jobs/${jobId}/result`;
            resultVideo.src = url;
            resultVideo.classList.remove("hidden");
            thumbnail.classList.add("hidden");
            downloadLink.href = url;
            downloadLink.classList.remove("hidden");
          } else if (data.status === "error") {
            clearInterval(pollTimer);
            jobStatusEl.textContent = "Error: " + (data.error || "unknown error");
            processBtn.disabled = false;
          }
        })
        .catch(() => {
          clearInterval(pollTimer);
          jobStatusEl.textContent = "Lost connection to job status.";
          processBtn.disabled = false;
        });
    }, 1000);
  }

  techniqueSelect.addEventListener("change", () => {
    const spec = currentSpec();
    buildControls(spec);
    updateAbout(spec);
    updateVisibility(spec);
  });
  processBtn.addEventListener("click", startJob);

  if (window.confirmBeforeNav) {
    const abandonMessage = "You'll lose this editing session and any results you haven't downloaded. Continue?";
    window.confirmBeforeNav(document.getElementById("home-link"), abandonMessage);
    window.confirmBeforeNav(document.getElementById("new-upload-link"), abandonMessage);
  }

  buildControls(currentSpec());
  updateAbout(currentSpec());
  updateVisibility(currentSpec());
})();
