const LABEL_COLOR_VARS = {
  loaded_language: "--loaded_language",
  name_calling: "--name_calling",
  exaggeration_minimization: "--exaggeration_minimization",
  appeal_to_fear: "--appeal_to_fear",
  unsupported_claim: "--unsupported_claim",
  factual_neutral: "--factual_neutral",
};

const textArea = document.getElementById("input-text");
const analyzeBtn = document.getElementById("analyze-btn");
const outputDiv = document.getElementById("output");
const summaryDiv = document.getElementById("summary");

function colorFor(labelName) {
  const varName = LABEL_COLOR_VARS[labelName] || "--factual_neutral";
  return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
}

function hexToRgba(hex, alpha) {
  if (!hex || hex === "transparent") return "transparent";
  const clean = hex.replace("#", "");
  const bigint = parseInt(clean, 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function renderResults(results) {
  outputDiv.innerHTML = "";
  let flaggedCount = 0;

  results.forEach((item) => {
    const span = document.createElement("span");
    span.className = "sentence";
    span.textContent = item.sentence + " ";

    if (item.labels.length > 0) {
      flaggedCount += 1;
      const top = item.labels[0];
      span.style.backgroundColor = hexToRgba(colorFor(top.name), 0.35);
      const labelSummary = item.labels
        .map((l) => `${l.name.replace(/_/g, " ")} (${(l.score * 100).toFixed(0)}%)`)
        .join(", ");
      span.title = labelSummary;
    }
    outputDiv.appendChild(span);
  });

  const total = results.length;
  const neutralPct = total ? Math.round(((total - flaggedCount) / total) * 100) : 0;
  const flaggedPct = 100 - neutralPct;
  summaryDiv.hidden = false;
  summaryDiv.textContent = `${total} sentence${total === 1 ? "" : "s"} analyzed - ${neutralPct}% neutral, ${flaggedPct}% flagged with at least one technique.`;
}

async function analyze() {
  const text = textArea.value.trim();
  if (!text) return;

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing...";
  outputDiv.textContent = "";
  summaryDiv.hidden = true;

  try {
    const resp = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.error || `Request failed (${resp.status})`);
    }
    const results = await resp.json();
    renderResults(results);
  } catch (e) {
    outputDiv.textContent = `Error: ${e.message}`;
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze";
  }
}

analyzeBtn.addEventListener("click", analyze);
