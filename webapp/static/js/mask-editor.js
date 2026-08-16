(function () {
  const canvas = document.getElementById("mask-canvas");
  const previewImg = document.getElementById("preview-image");
  const brushSizeInput = document.getElementById("brush-size");
  const clearBtn = document.getElementById("clear-mask");
  if (!canvas || !previewImg) return;

  const ctx = canvas.getContext("2d");
  let enabled = false;
  let drawing = false;

  function resizeCanvasToImage() {
    const rect = previewImg.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const prevData = canvas.width && canvas.height ? canvas.toDataURL() : null;
    canvas.width = previewImg.naturalWidth || rect.width;
    canvas.height = previewImg.naturalHeight || rect.height;
    canvas.style.width = rect.width + "px";
    canvas.style.height = rect.height + "px";
    canvas.style.left = previewImg.offsetLeft + "px";
    canvas.style.top = previewImg.offsetTop + "px";
    // best-effort: canvas gets cleared on resize; a fresh mask per preview size is fine for v1
    void prevData;
  }

  function pointerPos(evt) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const clientX = evt.touches ? evt.touches[0].clientX : evt.clientX;
    const clientY = evt.touches ? evt.touches[0].clientY : evt.clientY;
    return { x: (clientX - rect.left) * scaleX, y: (clientY - rect.top) * scaleY };
  }

  function drawAt(x, y) {
    const brush = Number((brushSizeInput && brushSizeInput.value) || 24);
    ctx.fillStyle = "white";
    ctx.beginPath();
    ctx.arc(x, y, brush / 2, 0, Math.PI * 2);
    ctx.fill();
  }

  function startDraw(evt) {
    if (!enabled) return;
    drawing = true;
    const pos = pointerPos(evt);
    drawAt(pos.x, pos.y);
    evt.preventDefault();
  }

  function moveDraw(evt) {
    if (!enabled || !drawing) return;
    const pos = pointerPos(evt);
    drawAt(pos.x, pos.y);
    evt.preventDefault();
  }

  function endDraw() {
    if (!drawing) return;
    drawing = false;
    window.dispatchEvent(new Event("mask-updated"));
  }

  canvas.addEventListener("mousedown", startDraw);
  canvas.addEventListener("mousemove", moveDraw);
  window.addEventListener("mouseup", endDraw);
  canvas.addEventListener("touchstart", startDraw, { passive: false });
  canvas.addEventListener("touchmove", moveDraw, { passive: false });
  canvas.addEventListener("touchend", endDraw);

  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      window.dispatchEvent(new Event("mask-updated"));
    });
  }

  previewImg.addEventListener("load", resizeCanvasToImage);
  window.addEventListener("resize", resizeCanvasToImage);

  window.MaskEditor = {
    setEnabled(value) {
      enabled = value;
      canvas.classList.toggle("hidden", !value);
    },
    getDataURL() {
      if (!canvas.width || !canvas.height) return null;
      const blank = document.createElement("canvas");
      blank.width = canvas.width;
      blank.height = canvas.height;
      if (canvas.toDataURL() === blank.toDataURL()) return null;
      return canvas.toDataURL("image/png");
    },
    clear() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    },
  };

  resizeCanvasToImage();
})();
