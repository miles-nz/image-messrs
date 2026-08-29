(function () {
  const overlay = document.getElementById("crop-overlay");
  const rectEl = document.getElementById("crop-rect");
  const previewImg = document.getElementById("preview-image");
  if (!overlay || !rectEl || !previewImg) return;

  const MIN_SIZE = 24; // px, in overlay-local coordinates

  let enabled = false;
  let aspect = null; // width/height ratio, or null for free
  let normRect = null; // {x,y,width,height} fractions of the displayed image, or null

  let activeHandle = null; // "move" | "draw" | "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w"
  let dragAnchor = { x: 0, y: 0 }; // overlay-local px; the fixed point a resize pivots around
  let dragMoveOffset = { x: 0, y: 0 }; // for "move": pointer offset from the rect's top-left
  let liveRectPx = null; // {left, top, width, height} in overlay-local px, during a drag

  function overlaySize() {
    return { w: overlay.clientWidth, h: overlay.clientHeight };
  }

  function pxFromNorm(norm) {
    const { w, h } = overlaySize();
    return { left: norm.x * w, top: norm.y * h, width: norm.width * w, height: norm.height * h };
  }

  function normFromPx(px) {
    const { w, h } = overlaySize();
    if (!w || !h) return null;
    return { x: px.left / w, y: px.top / h, width: px.width / w, height: px.height / h };
  }

  function renderRect(px) {
    if (!px) {
      rectEl.classList.add("hidden");
      return;
    }
    rectEl.classList.remove("hidden");
    rectEl.style.left = px.left + "px";
    rectEl.style.top = px.top + "px";
    rectEl.style.width = px.width + "px";
    rectEl.style.height = px.height + "px";
  }

  function resizeOverlayToImage() {
    const rect = previewImg.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    overlay.style.left = previewImg.offsetLeft + "px";
    overlay.style.top = previewImg.offsetTop + "px";
    overlay.style.width = rect.width + "px";
    overlay.style.height = rect.height + "px";
    renderRect(normRect ? pxFromNorm(normRect) : null);
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function localPoint(evt) {
    const rect = overlay.getBoundingClientRect();
    const clientX = evt.touches ? evt.touches[0].clientX : evt.clientX;
    const clientY = evt.touches ? evt.touches[0].clientY : evt.clientY;
    const { w, h } = overlaySize();
    return { x: clamp(clientX - rect.left, 0, w), y: clamp(clientY - rect.top, 0, h) };
  }

  function isFreeX(handle) {
    return handle !== "move" && handle !== "n" && handle !== "s";
  }
  function isFreeY(handle) {
    return handle !== "move" && handle !== "e" && handle !== "w";
  }

  function anchorForHandle(handle, rectPx) {
    const right = rectPx.left + rectPx.width;
    const bottom = rectPx.top + rectPx.height;
    switch (handle) {
      case "nw":
        return { x: right, y: bottom };
      case "n":
        return { x: rectPx.left, y: bottom };
      case "ne":
        return { x: rectPx.left, y: bottom };
      case "e":
        return { x: rectPx.left, y: rectPx.top };
      case "se":
        return { x: rectPx.left, y: rectPx.top };
      case "s":
        return { x: rectPx.left, y: rectPx.top };
      case "sw":
        return { x: right, y: rectPx.top };
      case "w":
        return { x: right, y: rectPx.top };
      default:
        return { x: rectPx.left, y: rectPx.top };
    }
  }

  function updateDrag(point) {
    const { w: overlayW, h: overlayH } = overlaySize();

    if (activeHandle === "move") {
      let left = clamp(point.x - dragMoveOffset.x, 0, overlayW - liveRectPx.width);
      let top = clamp(point.y - dragMoveOffset.y, 0, overlayH - liveRectPx.height);
      liveRectPx = { left, top, width: liveRectPx.width, height: liveRectPx.height };
      renderRect(liveRectPx);
      return;
    }

    const base = liveRectPx;
    const freeX = isFreeX(activeHandle);
    const freeY = isFreeY(activeHandle);

    let left = base.left;
    let width = base.width;
    let top = base.top;
    let height = base.height;

    if (freeX) {
      const dirX = point.x >= dragAnchor.x ? 1 : -1;
      width = Math.abs(point.x - dragAnchor.x);
      left = dirX > 0 ? dragAnchor.x : dragAnchor.x - width;
    }
    if (freeY) {
      const dirY = point.y >= dragAnchor.y ? 1 : -1;
      height = Math.abs(point.y - dragAnchor.y);
      top = dirY > 0 ? dragAnchor.y : dragAnchor.y - height;
    }

    if (aspect) {
      if (freeX && !freeY) {
        height = width / aspect;
        top = base.top;
      } else if (freeY && !freeX) {
        width = height * aspect;
        left = base.left;
      } else if (freeX && freeY) {
        height = width / aspect;
        const dirY = point.y >= dragAnchor.y ? 1 : -1;
        top = dirY > 0 ? dragAnchor.y : dragAnchor.y - height;
      }
    }

    if (left < 0) {
      width += left;
      left = 0;
    }
    if (top < 0) {
      height += top;
      top = 0;
    }
    if (left + width > overlayW) width = overlayW - left;
    if (top + height > overlayH) height = overlayH - top;

    if (width < MIN_SIZE || height < MIN_SIZE) return; // ignore degenerate frames, keep last good rect

    liveRectPx = { left, top, width, height };
    renderRect(liveRectPx);
  }

  function handleDragStart(handle, evt) {
    if (!enabled) return;
    const point = localPoint(evt);
    activeHandle = handle;

    if (handle === "move") {
      liveRectPx = pxFromNorm(normRect);
      dragMoveOffset = { x: point.x - liveRectPx.left, y: point.y - liveRectPx.top };
    } else if (handle === "draw") {
      liveRectPx = { left: point.x, top: point.y, width: 0, height: 0 };
      dragAnchor = { x: point.x, y: point.y };
    } else {
      liveRectPx = pxFromNorm(normRect);
      dragAnchor = anchorForHandle(handle, liveRectPx);
    }
    evt.preventDefault();
  }

  function endDrag() {
    if (!activeHandle) return;
    activeHandle = null;
    if (!liveRectPx || liveRectPx.width < MIN_SIZE || liveRectPx.height < MIN_SIZE) {
      liveRectPx = null;
      renderRect(normRect ? pxFromNorm(normRect) : null);
      return;
    }
    const norm = normFromPx(liveRectPx);
    liveRectPx = null;
    if (!norm) return;
    normRect = norm;
    window.dispatchEvent(new CustomEvent("crop-updated", { detail: normRect }));
  }

  overlay.addEventListener("mousedown", (evt) => {
    if (evt.target.closest(".crop-handle")) return;
    handleDragStart(evt.target === rectEl ? "move" : "draw", evt);
  });
  overlay.addEventListener(
    "touchstart",
    (evt) => {
      if (evt.target.closest(".crop-handle")) return;
      handleDragStart(evt.target === rectEl ? "move" : "draw", evt);
    },
    { passive: false }
  );

  rectEl.querySelectorAll(".crop-handle").forEach((handleEl) => {
    const start = (evt) => {
      evt.stopPropagation();
      handleDragStart(handleEl.dataset.handle, evt);
    };
    handleEl.addEventListener("mousedown", start);
    handleEl.addEventListener("touchstart", start, { passive: false });
  });

  window.addEventListener("mousemove", (evt) => {
    if (!activeHandle) return;
    updateDrag(localPoint(evt));
  });
  window.addEventListener(
    "touchmove",
    (evt) => {
      if (!activeHandle) return;
      updateDrag(localPoint(evt));
      evt.preventDefault();
    },
    { passive: false }
  );
  window.addEventListener("mouseup", endDrag);
  window.addEventListener("touchend", endDrag);

  previewImg.addEventListener("load", () => {
    if (enabled) resizeOverlayToImage();
  });
  window.addEventListener("resize", () => {
    if (enabled) resizeOverlayToImage();
  });

  window.CropEditor = {
    setEnabled(value) {
      enabled = value;
      overlay.classList.toggle("hidden", !value);
      if (value) resizeOverlayToImage();
    },
    setAspect(value) {
      if (!value) {
        aspect = null;
        return;
      }
      const [w, h] = value.split(":").map(Number);
      aspect = w && h ? w / h : null;
    },
    reset() {
      normRect = null;
      renderRect(null);
      window.dispatchEvent(new CustomEvent("crop-updated", { detail: null }));
    },
    hasCrop() {
      return normRect !== null;
    },
  };
})();
