// Vanilla-JS/WebGL2 port of the React Bits "WarpText" component (no ogl,
// no build pipeline - see [[feedback_reactbits_copy_for_ai]] in memory for
// why this project always ports to raw DOM/WebGL instead of npm-installing
// the source). Renders text to a 2D canvas, uploads it as a texture, and
// warps it with a fragment shader (ambient fbm drift + pointer bulge/ripple
// + chromatic split). Auto-instantiates on any `[data-warp-text]` element.

(() => {
    const VERTEX_SRC = `#version 300 es
in vec2 position;
in vec2 uv;
out vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position, 0.0, 1.0);
}
`;

    const FRAGMENT_SRC = `#version 300 es
precision highp float;

uniform sampler2D uTextTexture;
uniform vec2 uResolution;
uniform vec2 uPointer;
uniform float uPointerActive;
uniform float uTime;
uniform float uWarpStrength;
uniform float uWarpScale;
uniform float uSpeed;
uniform float uPointerInfluence;
uniform float uPointerStrength;
uniform float uRefraction;
uniform float uRipple;
uniform float uMotion;

in vec2 vUv;
out vec4 fragColor;

float hash(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);

  float a = hash(i);
  float b = hash(i + vec2(1.0, 0.0));
  float c = hash(i + vec2(0.0, 1.0));
  float d = hash(i + vec2(1.0, 1.0));

  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm(vec2 p) {
  float value = 0.0;
  float amplitude = 0.5;
  for (int i = 0; i < 4; i++) {
    value += amplitude * noise(p);
    p *= 2.02;
    amplitude *= 0.5;
  }
  return value;
}

vec4 sampleText(vec2 uv) {
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
    return vec4(0.0);
  }
  return texture(uTextTexture, uv);
}

void main() {
  vec2 uv = vUv;
  float aspect = uResolution.x / max(uResolution.y, 1.0);
  float time = uTime * uSpeed;
  float scale = max(uWarpScale, 0.001);

  vec2 drift = vec2(time * 0.055, -time * 0.045);
  float n1 = fbm(uv * scale * 3.1 + drift);
  float n2 = fbm((uv + 19.17) * scale * 3.4 - drift.yx);
  vec2 ambient = (vec2(n1, n2) - 0.5) * uWarpStrength * 0.045 * uMotion;

  vec2 pointerDelta = uv - uPointer;
  vec2 aspectDelta = vec2(pointerDelta.x * aspect, pointerDelta.y);
  float dist = length(aspectDelta);
  float radius = max(uPointerInfluence, 0.001);
  float t = clamp(dist / radius, 0.0, 1.0);
  float lens = smoothstep(radius, 0.0, dist) * uPointerActive;
  float bulge = t * (1.0 - t) * (1.0 - t) * 6.75 * uPointerActive;
  vec2 dir = dist > 0.0001 ? vec2(aspectDelta.x / aspect, aspectDelta.y) / dist : vec2(0.0);

  float rippleWave = sin(dist * 28.0 - time * 4.2) * 0.5 + 0.5;
  float rippleRing = (rippleWave - 0.5) * uRipple;
  vec2 pointerWarp = -dir * bulge * uPointerStrength * 0.045;
  pointerWarp += dir * rippleRing * bulge * uPointerStrength * 0.016;

  vec2 displaced = uv + ambient + pointerWarp;
  vec2 splitDir = ambient + pointerWarp;
  float splitLen = length(splitDir);
  splitDir = splitLen > 0.00001 ? splitDir / splitLen : vec2(0.7071, 0.7071);
  vec2 split = splitDir * uRefraction * 0.16 * (0.35 + lens * 1.65);

  vec4 base = sampleText(displaced);
  float r = sampleText(displaced + split).r;
  float g = base.g;
  float b = sampleText(displaced - split).b;
  float a = max(max(sampleText(displaced + split).a, base.a), sampleText(displaced - split).a);

  vec3 color = vec3(r, g, b) + lens * base.a * 0.055;
  fragColor = vec4(color, a);
}
`;

    function compileShader(gl, type, source) {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            const info = gl.getShaderInfoLog(shader);
            gl.deleteShader(shader);
            throw new Error(`WarpText shader compile error: ${info}`);
        }
        return shader;
    }

    function createProgram(gl, vertexSrc, fragmentSrc) {
        const vertexShader = compileShader(gl, gl.VERTEX_SHADER, vertexSrc);
        const fragmentShader = compileShader(
            gl,
            gl.FRAGMENT_SHADER,
            fragmentSrc,
        );
        const program = gl.createProgram();
        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);
        gl.linkProgram(program);
        gl.deleteShader(vertexShader);
        gl.deleteShader(fragmentShader);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            const info = gl.getProgramInfoLog(program);
            gl.deleteProgram(program);
            throw new Error(`WarpText program link error: ${info}`);
        }
        return program;
    }

    function getFontValue(value) {
        return typeof value === "number" ? `${value}px` : value;
    }

    function measureLine(ctx, line, letterSpacing) {
        const chars = Array.from(line);
        const textWidth = chars.reduce(
            (width, char) => width + ctx.measureText(char).width,
            0,
        );
        return textWidth + Math.max(0, chars.length - 1) * letterSpacing;
    }

    function drawLine(ctx, line, x, y, letterSpacing) {
        const chars = Array.from(line);
        let cursor = x - measureLine(ctx, line, letterSpacing) / 2;
        chars.forEach((char, index) => {
            ctx.fillText(char, cursor, y);
            cursor +=
                ctx.measureText(char).width +
                (index === chars.length - 1 ? 0 : letterSpacing);
        });
    }

    // Resolves font shorthand (e.g. fontFamily: "inherit") against a hidden
    // probe placed in the real DOM, same technique the source React component
    // uses so the rasterized text matches what CSS would actually render.
    function resolveFontMetrics(container, props) {
        const probe = document.createElement("span");
        probe.textContent = props.text;
        Object.assign(probe.style, {
            position: "absolute",
            visibility: "hidden",
            pointerEvents: "none",
            whiteSpace: "pre",
            inset: "0 auto auto 0",
            fontFamily: props.fontFamily,
            fontSize: getFontValue(props.fontSize),
            fontWeight: String(props.fontWeight),
            letterSpacing: getFontValue(props.letterSpacing),
            lineHeight:
                typeof props.lineHeight === "number"
                    ? String(props.lineHeight)
                    : props.lineHeight,
        });
        container.appendChild(probe);
        const computed = window.getComputedStyle(probe);
        let fontSizePx = parseFloat(computed.fontSize) || 96;
        const fontFamily = computed.fontFamily || "sans-serif";
        const fontWeight = computed.fontWeight || String(props.fontWeight);
        let letterSpacing =
            computed.letterSpacing === "normal"
                ? 0
                : parseFloat(computed.letterSpacing) || 0;
        let lineHeight = parseFloat(computed.lineHeight);
        if (!Number.isFinite(lineHeight)) {
            lineHeight =
                fontSizePx *
                (typeof props.lineHeight === "number"
                    ? props.lineHeight
                    : 0.92);
        }
        const measuredWidth = probe.getBoundingClientRect().width;
        probe.remove();
        return {
            fontSizePx,
            fontFamily,
            fontWeight,
            letterSpacing,
            lineHeight,
            measuredWidth,
        };
    }

    function buildTextCanvas(width, height, dpr, props, metrics) {
        const canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.floor(width * dpr));
        canvas.height = Math.max(1, Math.floor(height * dpr));
        const ctx = canvas.getContext("2d");
        if (!ctx) return canvas;

        let { fontSizePx, fontFamily, fontWeight, letterSpacing, lineHeight } =
            metrics;

        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, width, height);
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillStyle = props.color;
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = "high";

        const lines = String(props.text || "").split("\n");
        const applyFont = () => {
            ctx.font = `${fontWeight} ${fontSizePx}px ${fontFamily}`;
        };
        applyFont();

        const maxWidth = width * 0.86;
        const maxHeight = height * 0.78;
        const widest = Math.max(
            ...lines.map((line) => measureLine(ctx, line, letterSpacing)),
            1,
        );
        const blockHeight = Math.max(lineHeight * lines.length, 1);
        const fit = Math.min(1, maxWidth / widest, maxHeight / blockHeight);

        if (fit < 1) {
            fontSizePx *= fit;
            letterSpacing *= fit;
            lineHeight *= fit;
            applyFont();
        }

        const startY = height / 2 - (lineHeight * (lines.length - 1)) / 2;
        lines.forEach((line, index) =>
            drawLine(
                ctx,
                line,
                width / 2,
                startY + index * lineHeight,
                letterSpacing,
            ),
        );

        return canvas;
    }

    function initWarpText(container, props) {
        if (!container || typeof window === "undefined") return () => {};

        const canvas = document.createElement("canvas");
        canvas.style.position = "absolute";
        canvas.style.display = "block";
        canvas.setAttribute("aria-hidden", "true");

        const gl = canvas.getContext("webgl2", {
            alpha: true,
            premultipliedAlpha: false,
            antialias: true,
        });
        if (!gl) {
            console.warn("WarpText: WebGL2 is not available.");
            return () => {};
        }

        let program;
        try {
            program = createProgram(gl, VERTEX_SRC, FRAGMENT_SRC);
        } catch (error) {
            console.warn(error.message);
            return () => {};
        }

        gl.clearColor(0, 0, 0, 0);

        // One oversized triangle covering clip space - cheaper than a quad and
        // avoids a diagonal seam.
        const positionBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
        gl.bufferData(
            gl.ARRAY_BUFFER,
            new Float32Array([-1, -1, 3, -1, -1, 3]),
            gl.STATIC_DRAW,
        );

        const uvBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, uvBuffer);
        gl.bufferData(
            gl.ARRAY_BUFFER,
            new Float32Array([0, 0, 2, 0, 0, 2]),
            gl.STATIC_DRAW,
        );

        const vao = gl.createVertexArray();
        gl.bindVertexArray(vao);
        const positionLoc = gl.getAttribLocation(program, "position");
        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
        gl.enableVertexAttribArray(positionLoc);
        gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);
        const uvLoc = gl.getAttribLocation(program, "uv");
        gl.bindBuffer(gl.ARRAY_BUFFER, uvBuffer);
        gl.enableVertexAttribArray(uvLoc);
        gl.vertexAttribPointer(uvLoc, 2, gl.FLOAT, false, 0, 0);
        gl.bindVertexArray(null);

        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        const uniforms = {};
        [
            "uTextTexture",
            "uResolution",
            "uPointer",
            "uPointerActive",
            "uTime",
            "uWarpStrength",
            "uWarpScale",
            "uSpeed",
            "uPointerInfluence",
            "uPointerStrength",
            "uRefraction",
            "uRipple",
            "uMotion",
        ].forEach((name) => {
            uniforms[name] = gl.getUniformLocation(program, name);
        });

        let reduceMotion =
            window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ??
            false;
        const pointer = {
            x: 0.5,
            y: 0.5,
            tx: 0.5,
            ty: 0.5,
            active: 0,
            activeTarget: 0,
        };
        const startTime = performance.now();

        let disposed = false;
        let contextLost = false;
        let visible = true;
        let pageVisible = !document.hidden;
        let raf = 0;
        let resolution = [1, 1];

        function renderOnce() {
            if (disposed || contextLost) return;
            gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.useProgram(program);
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, texture);
            gl.uniform1i(uniforms.uTextTexture, 0);
            gl.uniform2fv(uniforms.uResolution, resolution);
            gl.uniform2f(uniforms.uPointer, pointer.x, pointer.y);
            gl.uniform1f(
                uniforms.uPointerActive,
                reduceMotion ? pointer.active * 0.35 : pointer.active,
            );
            gl.uniform1f(
                uniforms.uTime,
                reduceMotion ? 0 : (performance.now() - startTime) * 0.001,
            );
            gl.uniform1f(uniforms.uWarpStrength, props.warpStrength);
            gl.uniform1f(uniforms.uWarpScale, props.warpScale);
            gl.uniform1f(uniforms.uSpeed, props.speed);
            gl.uniform1f(uniforms.uPointerInfluence, props.pointerInfluence);
            gl.uniform1f(uniforms.uPointerStrength, props.pointerStrength);
            gl.uniform1f(uniforms.uRefraction, props.refraction);
            gl.uniform1f(uniforms.uRipple, props.ripple ? 1 : 0);
            gl.uniform1f(uniforms.uMotion, reduceMotion ? 0 : 1);
            gl.enable(gl.BLEND);
            gl.blendFuncSeparate(
                gl.SRC_ALPHA,
                gl.ONE_MINUS_SRC_ALPHA,
                gl.ONE,
                gl.ONE_MINUS_SRC_ALPHA,
            );
            gl.bindVertexArray(vao);
            gl.drawArrays(gl.TRIANGLES, 0, 3);
            gl.bindVertexArray(null);
        }

        // Sizes the *container* to the text's tight natural box - the same
        // box a normal inline text flow (e.g. masked-heading.js) would
        // occupy - so this drops into a layout exactly like any other
        // header wordmark, no extra padding thrown in around it.
        //
        // buildTextCanvas keeps glyphs under 86%/78% of whatever box it's
        // given (headroom for warp displacement + anti-aliasing at the
        // edges) and shrinks them to fit otherwise. Rather than shrinking
        // the visible text to make room for that, or inflating the
        // container to dodge it (which is what made this wider than
        // masked-heading in the first place), give the *canvas* a bleed
        // margin past the container's tight box - the layout footprint
        // stays exact, the render buffer still gets its padding.
        function layoutBox(metrics) {
            // measuredWidth comes straight off a real DOM text node (see
            // resolveFontMetrics) rather than canvas measureText, so this
            // matches masked-heading.js's own (native text flow) box
            // exactly instead of approximating it.
            const lines = String(props.text || "").split("\n");
            const boxWidth = Math.ceil(Math.max(metrics.measuredWidth, 1));
            const boxHeight = Math.ceil(
                Math.max(metrics.lineHeight * lines.length, 1),
            );
            container.style.width = `${boxWidth}px`;
            container.style.height = `${boxHeight}px`;

            const canvasWidth = boxWidth / 0.86;
            const canvasHeight = boxHeight / 0.78;
            const bleedX = (canvasWidth - boxWidth) / 2;
            const bleedY = (canvasHeight - boxHeight) / 2;
            canvas.style.left = `${-bleedX}px`;
            canvas.style.top = `${-bleedY}px`;
            canvas.style.width = `${canvasWidth}px`;
            canvas.style.height = `${canvasHeight}px`;

            return { canvasWidth, canvasHeight };
        }

        function rasterize() {
            if (disposed || contextLost) return;
            const metrics = resolveFontMetrics(container, props);
            const { canvasWidth, canvasHeight } = layoutBox(metrics);
            if (canvasWidth <= 0 || canvasHeight <= 0) return;
            const dpr = Math.min(window.devicePixelRatio || 1, 2);
            canvas.width = Math.max(1, Math.floor(canvasWidth * dpr));
            canvas.height = Math.max(1, Math.floor(canvasHeight * dpr));
            resolution = [canvas.width, canvas.height];

            const textCanvas = buildTextCanvas(
                canvasWidth,
                canvasHeight,
                dpr,
                props,
                metrics,
            );
            gl.bindTexture(gl.TEXTURE_2D, texture);
            gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
            gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
            gl.texImage2D(
                gl.TEXTURE_2D,
                0,
                gl.RGBA,
                gl.RGBA,
                gl.UNSIGNED_BYTE,
                textCanvas,
            );
            renderOnce();
        }

        function onPointerMove(event) {
            if (event.pointerType === "touch") return;
            const rect = canvas.getBoundingClientRect();
            if (rect.width <= 0 || rect.height <= 0) return;
            pointer.tx = (event.clientX - rect.left) / rect.width;
            pointer.ty = 1 - (event.clientY - rect.top) / rect.height;
            pointer.activeTarget = 1;
        }

        function onPointerLeave() {
            pointer.activeTarget = 0;
        }

        function onContextLost(event) {
            event.preventDefault();
            contextLost = true;
            if (raf) cancelAnimationFrame(raf);
            raf = 0;
        }

        function onVisibility() {
            pageVisible = !document.hidden;
            if (pageVisible && visible && !raf)
                raf = requestAnimationFrame(loop);
            if (!pageVisible && raf) {
                cancelAnimationFrame(raf);
                raf = 0;
            }
        }

        const mediaQuery = window.matchMedia?.(
            "(prefers-reduced-motion: reduce)",
        );
        function onReducedMotion(event) {
            reduceMotion = event.matches;
            renderOnce();
        }

        function loop(now) {
            if (disposed || contextLost) return;
            const elapsed = (now - startTime) * 0.001;
            const idleX = 0.5 + Math.sin(elapsed * 0.33) * 0.12;
            const idleY = 0.5 + Math.cos(elapsed * 0.27) * 0.1;
            const targetX = pointer.activeTarget > 0 ? pointer.tx : idleX;
            const targetY = pointer.activeTarget > 0 ? pointer.ty : idleY;
            const damping = pointer.activeTarget > 0 ? 0.12 : 0.035;

            pointer.x += (targetX - pointer.x) * damping;
            pointer.y += (targetY - pointer.y) * damping;
            pointer.active +=
                ((pointer.activeTarget > 0 ? 1 : 0.18) - pointer.active) * 0.06;

            renderOnce();
            raf = requestAnimationFrame(loop);
        }

        const resizeObserver = new ResizeObserver(rasterize);
        const intersectionObserver = new IntersectionObserver(
            ([entry]) => {
                visible = entry.isIntersecting;
                if (visible && pageVisible && !raf)
                    raf = requestAnimationFrame(loop);
                if (!visible && raf) {
                    cancelAnimationFrame(raf);
                    raf = 0;
                }
            },
            { threshold: 0 },
        );

        canvas.addEventListener("pointermove", onPointerMove);
        canvas.addEventListener("pointerleave", onPointerLeave);
        canvas.addEventListener("webglcontextlost", onContextLost, false);
        document.addEventListener("visibilitychange", onVisibility);
        mediaQuery?.addEventListener("change", onReducedMotion);

        (async () => {
            if (document.fonts?.ready) {
                try {
                    await document.fonts.ready;
                } catch {}
            }
            if (disposed) return;
            container.appendChild(canvas);
            resizeObserver.observe(container);
            intersectionObserver.observe(container);
            rasterize();
            raf = requestAnimationFrame(loop);
        })();

        return function destroy() {
            disposed = true;
            if (raf) cancelAnimationFrame(raf);
            resizeObserver.disconnect();
            intersectionObserver.disconnect();
            canvas.removeEventListener("pointermove", onPointerMove);
            canvas.removeEventListener("pointerleave", onPointerLeave);
            canvas.removeEventListener("webglcontextlost", onContextLost);
            document.removeEventListener("visibilitychange", onVisibility);
            mediaQuery?.removeEventListener("change", onReducedMotion);
            if (!contextLost) {
                gl.deleteTexture(texture);
                gl.deleteBuffer(positionBuffer);
                gl.deleteBuffer(uvBuffer);
                gl.deleteVertexArray(vao);
                gl.deleteProgram(program);
            }
            canvas.remove();
        };
    }

    // Instantiated on the site-header wordmark, trying it as a replacement
    // for .glitch-title. Font-size/family are pinned to the same treatment
    // masked-heading.js uses elsewhere in the header (2rem, the site's system
    // sans stack) rather than WarpText's own defaults - everything else below
    // is the tuned config from the design pass.
    document.querySelectorAll("[data-warp-text]").forEach((el) => {
        initWarpText(el, {
            text: el.dataset.warpText || el.textContent.trim(),
            color: "#f8f5ff",
            fontSize: "2rem",
            fontWeight: 600,
            fontFamily:
                "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            letterSpacing: "0.03em",
            lineHeight: 1,
            warpStrength: 0.1,
            warpScale: 2,
            speed: 0.75,
            pointerInfluence: 1,
            pointerStrength: 1,
            refraction: 0.1,
            ripple: false,
        });
    });
})();
