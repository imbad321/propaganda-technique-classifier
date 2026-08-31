(function () {
  const form = document.getElementById("label-form");
  if (!form) return;

  const pills = Array.from(document.querySelectorAll(".pill"));
  const descEl = document.getElementById("label-desc");

  function showDescription(pill) {
    const label = pill.dataset.label;
    if (descEl && typeof DESCRIPTIONS !== "undefined") {
      descEl.textContent = DESCRIPTIONS[label] || "";
    }
  }

  pills.forEach((pill) => {
    pill.addEventListener("mouseenter", () => showDescription(pill));
  });

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT") return;

    const n = parseInt(e.key, 10);
    if (n >= 1 && n <= pills.length) {
      const checkbox = pills[n - 1].querySelector("input");
      checkbox.checked = !checkbox.checked;
      showDescription(pills[n - 1]);
      e.preventDefault();
    } else if (e.key === "Enter") {
      form.requestSubmit();
      e.preventDefault();
    }
  });
})();
