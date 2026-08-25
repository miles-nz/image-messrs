(function () {
  function clamp(v, a, b) {
    return v < a ? a : v > b ? b : v;
  }

  function initMaskedHeading(el) {
    const src = el.dataset.mediaSrc;
    if (!src) return;
    const mediaType = el.dataset.mediaType || "image";
    const originalText = el.textContent;

    try {
      const words = originalText.trim().split(/\s+/).filter(Boolean);
      if (!words.length) return;

      el.textContent = "";

      const measure = document.createElement("span");
      measure.className = "masked-heading__measure";
      const wordEls = [];
      words.forEach((word, i) => {
        const wordEl = document.createElement("span");
        wordEl.className = "masked-heading__word";
        wordEl.textContent = word;
        measure.appendChild(wordEl);
        if (i < words.length - 1) measure.appendChild(document.createTextNode(" "));
        wordEls.push(wordEl);
      });
      el.appendChild(measure);

      const svgNS = "http://www.w3.org/2000/svg";
      const svg = document.createElementNS(svgNS, "svg");
      svg.setAttribute("class", "masked-heading__defs");
      svg.setAttribute("aria-hidden", "true");
      const defs = document.createElementNS(svgNS, "defs");
      const clipId = "mh-clip-" + Math.random().toString(36).slice(2, 9);
      const clipPath = document.createElementNS(svgNS, "clipPath");
      clipPath.setAttribute("id", clipId);
      clipPath.setAttribute("clipPathUnits", "userSpaceOnUse");
      const glyphEls = words.map((word) => {
        const textEl = document.createElementNS(svgNS, "text");
        textEl.textContent = word;
        clipPath.appendChild(textEl);
        return textEl;
      });
      defs.appendChild(clipPath);
      svg.appendChild(defs);
      el.appendChild(svg);

      const reveal = document.createElement("span");
      reveal.className = "masked-heading__reveal";
      const clip = document.createElement("span");
      clip.className = "masked-heading__clip";
      clip.style.clipPath = `url(#${clipId})`;
      const media = document.createElement("span");
      media.className = "masked-heading__media";

      let mediaEl;
      if (mediaType === "video") {
        mediaEl = document.createElement("video");
        // Safari's autoplay eligibility check looks at the `muted` content
        // attribute, not just the (unreflected) `.muted` IDL property - set
        // both, and before `src`, so it's seen before Safari evaluates autoplay.
        mediaEl.setAttribute("muted", "");
        mediaEl.setAttribute("playsinline", "");
        mediaEl.muted = true;
        mediaEl.autoplay = true;
        mediaEl.loop = true;
        mediaEl.playsInline = true;
        mediaEl.src = src;
      } else {
        mediaEl = document.createElement("img");
        mediaEl.src = src;
        mediaEl.alt = "";
        mediaEl.draggable = false;
      }
      mediaEl.className = "masked-heading__source";
      mediaEl.addEventListener("error", () => {
        el.textContent = originalText;
        el.classList.remove("masked-heading-active");
      });

      media.appendChild(mediaEl);
      clip.appendChild(media);
      reveal.appendChild(clip);
      el.appendChild(reveal);

      if (mediaType === "video") {
        mediaEl.addEventListener(
          "playing",
          () => {
            // Driven by the video's own "timeupdate" (media playback
            // progress) rather than requestAnimationFrame - rAF can be
            // throttled/suppressed independently of whether the video
            // itself is actually playing, which would otherwise leave
            // playbackRate stuck at its starting value.
            const rampMs = 900;
            const startRate = mediaEl.playbackRate;
            const start = performance.now();
            const onTimeUpdate = () => {
              const t = clamp((performance.now() - start) / rampMs, 0, 1);
              const eased = 1 - Math.pow(1 - t, 3);
              mediaEl.playbackRate = startRate + (1 - startRate) * eased;
              if (t >= 1) mediaEl.removeEventListener("timeupdate", onTimeUpdate);
            };
            mediaEl.addEventListener("timeupdate", onTimeUpdate);
          },
          { once: true }
        );

        // Calling play() before Safari has loaded enough to confirm the
        // muted-autoplay exemption applies gets rejected with NotAllowedError -
        // wait for loadedmetadata so it has what it needs to grant it.
        const tryPlay = () => {
          // Assigning `src` resets playbackRate to 1 via the media load
          // algorithm, so it has to be (re)applied here, after that's done,
          // rather than up front at element creation.
          mediaEl.playbackRate = 0.15;
          const playPromise = mediaEl.play();
          if (playPromise && typeof playPromise.catch === "function") playPromise.catch(() => {});
        };
        if (mediaEl.readyState >= 1) tryPlay();
        else mediaEl.addEventListener("loadedmetadata", tryPlay, { once: true });

        // Safari's autoplay policy can refuse muted playback outright with
        // no way around it from script - but it always allows play() from a
        // real user gesture, so fall back to starting on first interaction.
        const playOnGesture = () => {
          if (mediaEl.paused) tryPlay();
        };
        document.addEventListener("pointerdown", playOnGesture, { once: true });
        document.addEventListener("keydown", playOnGesture, { once: true });
      }

      el.classList.add("masked-heading-active");

      function sync() {
        const cs = window.getComputedStyle(el);
        wordEls.forEach((wordEl, i) => {
          const glyph = glyphEls[i];
          glyph.setAttribute("x", wordEl.offsetLeft);
          glyph.setAttribute("y", wordEl.offsetTop + wordEl.offsetHeight * 0.82);
          glyph.style.fontFamily = cs.fontFamily;
          glyph.style.fontSize = cs.fontSize;
          glyph.style.fontWeight = cs.fontWeight;
        });
      }

      const ro = new ResizeObserver(sync);
      ro.observe(el);
      if (document.fonts && document.fonts.ready) document.fonts.ready.then(sync).catch(() => {});
      sync();

      const fillScale = 1.7;
      media.style.transform = `scale(${fillScale})`;

      const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      if (prefersReducedMotion) return;

      const drift = 10;
      const parallax = mediaType === "video" ? 8 : 45;
      const offset = { x: 0, y: 0, tx: 0, ty: 0 };

      function place() {
        const width = el.clientWidth;
        const height = el.clientHeight;
        const maxX = ((fillScale - 1) / 2) * width;
        const maxY = ((fillScale - 1) / 2) * height;
        const x = clamp(offset.x, -maxX, maxX);
        const y = clamp(offset.y, -maxY, maxY);
        media.style.transform = `translate3d(${x.toFixed(2)}px, ${y.toFixed(2)}px, 0) scale(${fillScale})`;
      }

      let last = performance.now();
      let clock = 0;
      function frame(now) {
        const dt = Math.min(0.05, (now - last) / 1000);
        last = now;
        clock += dt;
        const dx = Math.sin(clock * 0.5) * drift;
        const dy = Math.cos(clock * 0.4) * drift * 0.6;
        const ease = 1 - Math.exp(-dt / 0.18);
        offset.x += (offset.tx + dx - offset.x) * ease;
        offset.y += (offset.ty + dy - offset.y) * ease;
        place();
        requestAnimationFrame(frame);
      }
      requestAnimationFrame(frame);

      window.addEventListener("pointermove", (e) => {
        const nx = (e.clientX / window.innerWidth) * 2 - 1;
        const ny = (e.clientY / window.innerHeight) * 2 - 1;
        offset.tx = clamp(nx, -1, 1) * -parallax;
        offset.ty = clamp(ny, -1, 1) * -parallax;
      });
      document.addEventListener("mouseleave", () => {
        offset.tx = 0;
        offset.ty = 0;
      });
    } catch (err) {
      console.error("[masked-heading] failed, falling back to plain text", err);
      el.textContent = originalText;
      el.classList.remove("masked-heading-active");
    }
  }

  document.querySelectorAll(".masked-heading").forEach(initMaskedHeading);

  window.MaskedHeading = {
    refresh(src) {
      document.querySelectorAll(".masked-heading-active .masked-heading__source").forEach((mediaEl) => {
        mediaEl.src = src;
      });
    },
  };
})();
