document.addEventListener("DOMContentLoaded", function() {
  var observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(m) {
      if (m.attributeName === "data-md-color-scheme") {
        var scheme = document.body.getAttribute("data-md-color-scheme");
        document.querySelectorAll(".jupyter-wrapper").forEach(function(c) {
          if (scheme === "slate") { c.classList.add("jp-theme-dark"); c.classList.remove("jp-theme-light"); }
          else { c.classList.add("jp-theme-light"); c.classList.remove("jp-theme-dark"); }
        });
      }
    });
  });
  observer.observe(document.body, { attributes: true });
});
