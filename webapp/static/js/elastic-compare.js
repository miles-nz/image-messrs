(function () {
    // Vanilla port of React Bits' ElasticSlider overflow/spring-back mechanic,
    // adapted from a numeric value slider to a positional divider: dragging
    // past the divider's bounds is allowed to visually overshoot (decayed so
    // it never exceeds maxOverflow), then eases back with a single
    // overshoot-and-settle bounce on release instead of a hand-rolled RAF spring.
    function decay(rawPx, max) {
        if (rawPx <= 0) return 0;
        return (max * rawPx) / (max + rawPx);
    }

    function attach(dividerEl, handleEl, opts) {
        const maxOverflow = (opts && opts.maxOverflow) || 14;
        let releaseTimer = null;

        function clearReleasing() {
            dividerEl.classList.remove("compare-divider--releasing");
            if (releaseTimer) {
                clearTimeout(releaseTimer);
                releaseTimer = null;
            }
        }

        function setOverflow(px) {
            const scale = 1 - Math.min(0.35, Math.abs(px) / maxOverflow / 3);
            dividerEl.style.setProperty("--overflow-px", px.toFixed(2) + "px");
            dividerEl.style.setProperty("--overflow-scale", scale.toFixed(3));
            if (handleEl) {
                const handleScale = 1 + Math.min(0.3, (Math.abs(px) / maxOverflow) * 0.3);
                handleEl.style.setProperty("--overflow-handle-scale", handleScale.toFixed(3));
            }
        }

        function update(signedOverflowPx) {
            clearReleasing();
            const magnitude = decay(Math.abs(signedOverflowPx), maxOverflow);
            setOverflow(signedOverflowPx < 0 ? -magnitude : magnitude);
        }

        function release() {
            dividerEl.classList.add("compare-divider--releasing");
            setOverflow(0);
            releaseTimer = setTimeout(clearReleasing, 450);
            dividerEl.addEventListener(
                "transitionend",
                () => {
                    clearReleasing();
                },
                { once: true }
            );
        }

        setOverflow(0);

        return { update, release };
    }

    window.ElasticCompare = { attach, decay };
})();
