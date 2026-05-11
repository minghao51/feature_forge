(() => {
  function setExpanded(container, button, expanded) {
    container.classList.toggle("expanded", expanded);
    button.textContent = expanded ? "Exit Fullscreen" : "Expand";
    button.setAttribute("aria-expanded", expanded ? "true" : "false");
    document.body.classList.toggle("notebook-fullscreen-active", expanded);
    if (expanded) {
      button.focus();
    }
  }

  function bindNotebookControls() {
    const buttons = document.querySelectorAll("[data-notebook-toggle]");
    for (const button of buttons) {
      button.addEventListener("click", () => {
        const container = button.closest(".iframe-container");
        if (!container) {
          return;
        }
        const expanded = !container.classList.contains("expanded");
        setExpanded(container, button, expanded);
      });
    }
  }

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") {
      return;
    }
    const expandedContainer = document.querySelector(".iframe-container.expanded");
    if (!expandedContainer) {
      return;
    }
    const button = expandedContainer.querySelector("[data-notebook-toggle]");
    if (!button) {
      return;
    }
    setExpanded(expandedContainer, button, false);
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bindNotebookControls);
  } else {
    bindNotebookControls();
  }
})();
