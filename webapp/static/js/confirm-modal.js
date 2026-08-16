(function () {
  function ensureModal() {
    let modal = document.getElementById("confirm-modal");
    if (modal) return modal;

    modal = document.createElement("div");
    modal.id = "confirm-modal";
    modal.className = "confirm-modal hidden";
    modal.innerHTML =
      '<div class="confirm-modal-backdrop"></div>' +
      '<div class="confirm-modal-dialog" role="dialog" aria-modal="true">' +
      '<h2 class="confirm-modal-title"></h2>' +
      '<p class="confirm-modal-message"></p>' +
      '<div class="confirm-modal-actions">' +
      '<button type="button" class="confirm-modal-cancel">Cancel</button>' +
      '<button type="button" class="confirm-modal-confirm">Leave</button>' +
      "</div>" +
      "</div>";
    document.body.appendChild(modal);
    return modal;
  }

  function showConfirmModal({ title, message, confirmLabel, onConfirm }) {
    const modal = ensureModal();
    modal.querySelector(".confirm-modal-title").textContent = title || "Abandon edit?";
    modal.querySelector(".confirm-modal-message").textContent = message || "";
    modal.querySelector(".confirm-modal-confirm").textContent = confirmLabel || "Leave";
    modal.classList.remove("hidden");

    const cancelBtn = modal.querySelector(".confirm-modal-cancel");
    const confirmBtn = modal.querySelector(".confirm-modal-confirm");
    const backdrop = modal.querySelector(".confirm-modal-backdrop");

    function close() {
      modal.classList.add("hidden");
      cancelBtn.removeEventListener("click", close);
      confirmBtn.removeEventListener("click", handleConfirm);
      backdrop.removeEventListener("click", close);
      document.removeEventListener("keydown", handleKey);
    }
    function handleConfirm() {
      close();
      if (onConfirm) onConfirm();
    }
    function handleKey(evt) {
      if (evt.key === "Escape") close();
    }

    cancelBtn.addEventListener("click", close);
    confirmBtn.addEventListener("click", handleConfirm);
    backdrop.addEventListener("click", close);
    document.addEventListener("keydown", handleKey);

    confirmBtn.focus();
  }

  function confirmBeforeNav(link, message) {
    if (!link) return;
    link.addEventListener("click", (evt) => {
      evt.preventDefault();
      showConfirmModal({
        message: message || "You'll lose this editing session. Continue?",
        confirmLabel: "Leave",
        onConfirm: () => {
          window.location.href = link.href;
        },
      });
    });
  }

  window.showConfirmModal = showConfirmModal;
  window.confirmBeforeNav = confirmBeforeNav;
})();
