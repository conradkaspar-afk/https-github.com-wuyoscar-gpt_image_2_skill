const map = L.map("map").setView([39.5, -98.35], 4);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 18,
  attribution: "© OpenStreetMap",
}).addTo(map);

let districtLayer = null;

async function loadStates() {
  const res = await fetch("/states");
  const { states } = await res.json();
  const sel = document.getElementById("state");
  for (const s of states) {
    const opt = document.createElement("option");
    opt.value = s;
    opt.textContent = s;
    sel.appendChild(opt);
  }
}

const stepsEl = document.getElementById("steps");
const stepsLabel = document.getElementById("stepsLabel");
stepsEl.addEventListener("input", () => (stepsLabel.textContent = stepsEl.value));

document.getElementById("go").addEventListener("click", async () => {
  const btn = document.getElementById("go");
  const status = document.getElementById("status");
  const metricsEl = document.getElementById("metrics");
  btn.disabled = true;
  metricsEl.innerHTML = "";
  status.textContent = "submitting…";

  const payload = {
    state: document.getElementById("state").value,
    num_districts: parseInt(document.getElementById("districts").value, 10),
    objective: document.getElementById("objective").value,
    steps: parseInt(stepsEl.value, 10),
  };

  const res = await fetch("/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    status.textContent = "error: " + (await res.text());
    btn.disabled = false;
    return;
  }
  const { job_id } = await res.json();

  while (true) {
    await new Promise((r) => setTimeout(r, 2000));
    const j = await (await fetch(`/job/${job_id}`)).json();
    status.textContent = `${j.status} — ${j.message} (${Math.round(j.progress * 100)}%)`;
    if (j.status === "done") {
      render(j.result);
      break;
    }
    if (j.status === "error") {
      status.textContent = "error: " + j.message;
      break;
    }
  }
  btn.disabled = false;
});

function render({ geojson, metrics }) {
  if (districtLayer) map.removeLayer(districtLayer);
  districtLayer = L.geoJSON(geojson, {
    style: (f) => ({
      color: "#333",
      weight: 1,
      fillColor: f.properties.party === "D" ? "#3b6ed8" : "#d83b3b",
      fillOpacity: 0.55,
    }),
    onEachFeature: (f, layer) => {
      const p = f.properties;
      layer.bindPopup(
        `<b>District ${p.district}</b><br>pop: ${Math.round(p.pop)}<br>D: ${Math.round(
          p.dem
        )} · R: ${Math.round(p.rep)}`
      );
    },
  }).addTo(map);
  map.fitBounds(districtLayer.getBounds());

  const m = metrics;
  document.getElementById("metrics").innerHTML = `
    <table>
      <tr><td>D seats</td><td>${m.dem_seats}</td></tr>
      <tr><td>R seats</td><td>${m.rep_seats}</td></tr>
      <tr><td>Efficiency gap</td><td>${m.efficiency_gap.toFixed(3)}</td></tr>
      <tr><td>Mean − median</td><td>${m.mean_median.toFixed(3)}</td></tr>
      <tr><td>Mean Polsby–Popper</td><td>${m.mean_polsby_popper.toFixed(3)}</td></tr>
      <tr><td>Pop deviation</td><td>${(m.population_deviation * 100).toFixed(2)}%</td></tr>
    </table>`;
}

loadStates();
