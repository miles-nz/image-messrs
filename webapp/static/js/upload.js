(function () {
  document.querySelectorAll(".dropzone").forEach((zone) => {
    const input = zone.querySelector('input[type="file"]');
    const label = zone.querySelector(".dropzone-filename");
    if (!input) return;

    input.addEventListener("change", () => {
      if (input.files.length && label) label.textContent = input.files[0].name;
    });

    zone.addEventListener("dragover", (evt) => {
      evt.preventDefault();
      zone.classList.add("dragover");
    });
    zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
    zone.addEventListener("drop", (evt) => {
      evt.preventDefault();
      zone.classList.remove("dragover");
      if (evt.dataTransfer.files.length) {
        input.files = evt.dataTransfer.files;
        if (label) label.textContent = evt.dataTransfer.files[0].name;
      }
    });
  });
})();
