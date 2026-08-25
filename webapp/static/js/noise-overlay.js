(function () {
  const canvas = document.getElementById("noise-overlay");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  if (!ctx) {
    canvas.remove();
    return;
  }

  const patternSize = 96;
  const patternAlpha = 22; // 0-255, kept low so it reads as texture, not static
  const refreshEveryNFrames = 2;

  const patternCanvas = document.createElement("canvas");
  patternCanvas.width = patternSize;
  patternCanvas.height = patternSize;
  const patternCtx = patternCanvas.getContext("2d");

  function regeneratePattern() {
    const imageData = patternCtx.createImageData(patternSize, patternSize);
    const data = imageData.data;
    for (let i = 0; i < data.length; i += 4) {
      const shade = (Math.random() * 255) | 0;
      data[i] = shade;
      data[i + 1] = shade;
      data[i + 2] = shade;
      data[i + 3] = patternAlpha;
    }
    patternCtx.putImageData(imageData, 0, 0);
  }

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    const width = Math.max(1, Math.floor(window.innerWidth * dpr));
    const height = Math.max(1, Math.floor(window.innerHeight * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
  }

  window.addEventListener("resize", resize);
  window.addEventListener("load", resize);

  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let frameCount = 0;

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const pattern = ctx.createPattern(patternCanvas, "repeat");
    ctx.fillStyle = pattern;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  function frame() {
    resize();
    if (frameCount % refreshEveryNFrames === 0) {
      regeneratePattern();
      draw();
    }
    frameCount++;
    if (!prefersReducedMotion) requestAnimationFrame(frame);
  }

  resize();
  regeneratePattern();
  draw();
  if (!prefersReducedMotion) requestAnimationFrame(frame);
})();
