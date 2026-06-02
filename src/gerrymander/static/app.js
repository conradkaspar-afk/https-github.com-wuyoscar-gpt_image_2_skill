(function () {
  "use strict";

  // ---- DOM handles --------------------------------------------------------
  const $state = document.getElementById("state-select");
  const $seats = document.getElementById("seats-slider");
  const $seatsDisplay = document.getElementById("seats-display");
  const $seatsHint = document.getElementById("seats-hint");
  const $intensity = document.getElementById("intensity-slider");
  const $intensityDisplay = document.getElementById("intensity-display");
  const $loading = document.getElementById("loading");
  const $hover = document.getElementById("hover");

  // ---- Leaflet map setup --------------------------------------------------
  const map = L.map("map", {
    center: [39.8, -98.5],
    zoom: 4,
    zoomControl: true,
    attributionControl: false,
  });

  // Subtle base layer (OSM tile via leaflet's default attribution-free style is N/A
  // offline). We instead draw on a plain background; the county polygons themselves
  // give the user enough geographic context.
  L.tileLayer("", { tileSize: 256, opacity: 0 });

  const stateLayerGroup = L.layerGroup().addTo(map);
  let activeStates = [];
  let currentFetchToken = 0;

  // ---- Helpers ------------------------------------------------------------
  function districtColor(districtId, totalDistricts, dPct) {
    // Hue cycles by district id; saturation tinted by party lean.
    const hue = (districtId * 137.508) % 360;
    let sat = 55;
    let light = 60;
    if (dPct > 52) { sat = 65; light = 56; }
    else if (dPct < 48) { sat = 65; light = 56; }
    else { sat = 35; light = 70; }
    return `hsl(${hue.toFixed(0)}, ${sat}%, ${light}%)`;
  }

  function partyColor(dPct) {
    if (dPct > 52) return "#2563eb";  // D
    if (dPct < 48) return "#dc2626";  // R
    return "#b89200";                 // tossup
  }

  function fmtPct(x) { return (x).toFixed(1) + "%"; }
  function fmtSigned(x) { return (x >= 0 ? "+" : "") + x.toFixed(2) + "%"; }
  function fmtInt(x) { return x.toLocaleString(); }

  // ---- API ----------------------------------------------------------------
  async function loadStates() {
    const r = await fetch("/api/states");
    const data = await r.json();
    activeStates = data;
    for (const s of data) {
      const opt = document.createElement("option");
      opt.value = s.code;
      opt.textContent = `${s.name} (${s.default_seats} seat${s.default_seats === 1 ? "" : "s"})`;
      $state.appendChild(opt);
    }
    $state.value = "PA";
    syncSlider();
  }

  function selectedStateMeta() {
    return activeStates.find((s) => s.code === $state.value) || null;
  }

  function syncSlider() {
    const s = selectedStateMeta();
    if (!s) return;
    $seats.min = String(s.min_seats);
    $seats.max = String(Math.max(s.max_seats, s.default_seats + 4));
    $seats.value = String(s.default_seats);
    $seatsDisplay.textContent = s.default_seats;
    $seatsHint.textContent = `${s.name}'s real congressional delegation: ${s.default_seats} seat${s.default_seats === 1 ? "" : "s"}.`;
  }

  async function refresh() {
    const token = ++currentFetchToken;
    const state = $state.value;
    const party = document.querySelector('input[name="party"]:checked').value;
    const seats = parseInt($seats.value, 10);
    const intensity = parseFloat($intensity.value);
    const url = `/api/plan?state=${encodeURIComponent(state)}&party=${party}&seats=${seats}&intensity=${intensity}`;
    $loading.classList.remove("hidden");
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error("API " + r.status);
      const data = await r.json();
      if (token !== currentFetchToken) return; // a newer call already supersedes us
      renderPlan(data);
    } catch (err) {
      console.error(err);
      alert("Failed to fetch plan: " + err.message);
    } finally {
      if (token === currentFetchToken) {
        $loading.classList.add("hidden");
      }
    }
  }

  // ---- Rendering ----------------------------------------------------------
  function renderPlan(data) {
    stateLayerGroup.clearLayers();

    const n = data.districts.length;
    const districtColors = data.districts.map((d) => districtColor(d.id, n, d.d_pct));

    const layer = L.geoJSON(data.geojson, {
      style: (feat) => {
        const did = feat.properties.district_id;
        return {
          color: "#0a0c10",          // district boundary stroke (overlay handles that thicker)
          weight: 0.4,
          opacity: 0.65,
          fillColor: districtColors[did] || "#cccccc",
          fillOpacity: 0.78,
        };
      },
      onEachFeature: (feat, lyr) => {
        const p = feat.properties;
        const dist = data.districts[p.district_id];
        lyr.on("mouseover", (e) => {
          lyr.setStyle({ weight: 1.4, color: "#000" });
          showHover(e, feat, dist);
        });
        lyr.on("mousemove", (e) => moveHover(e));
        lyr.on("mouseout", () => {
          layer.resetStyle(lyr);
          hideHover();
        });
        lyr.on("click", () => {
          map.fitBounds(lyr.getBounds(), { padding: [20, 20], maxZoom: 8 });
        });
      },
    }).addTo(stateLayerGroup);

    // Overlay: thick black lines between counties whose district_id differs.
    // We do this by drawing a second pass where we stroke the boundary of each
    // district as a union outline. For simplicity, we overlay every feature
    // outline but only show heavy stroke where neighboring features have
    // different districts — Leaflet doesn't expose adjacency, so we instead
    // emphasize by drawing each district polygon outline thick once. Simpler:
    // render each district as a separate filled layer with thick stroke.
    // Group features by district id:
    const byDistrict = {};
    data.geojson.features.forEach((f) => {
      const id = f.properties.district_id;
      if (!byDistrict[id]) byDistrict[id] = [];
      byDistrict[id].push(f);
    });
    Object.keys(byDistrict).forEach((id) => {
      const districtId = parseInt(id, 10);
      const fc = { type: "FeatureCollection", features: byDistrict[id] };
      L.geoJSON(fc, {
        style: () => ({
          color: "#0a0c10", weight: 2.2, opacity: 1,
          fillOpacity: 0, fill: false, interactive: false,
        }),
      }).addTo(stateLayerGroup);
    });

    // Fit to bbox.
    const b = data.bbox;
    map.fitBounds([[b[1], b[0]], [b[3], b[2]]], { padding: [20, 20] });

    // Update side panel.
    updateSummary(data);
  }

  function updateSummary(data) {
    document.getElementById("sum-state").textContent = data.state_name + " (" + data.state + ")";
    document.getElementById("sum-seats").textContent = data.seats;
    document.getElementById("sum-party").textContent =
      data.party === "neutral" ? "Neutral baseline" :
      (data.party === "D" ? "Democrat" : "Republican");
    document.getElementById("sum-d-vote").textContent = fmtPct(data.metrics.D_vote_share * 100);
    const dseats = Math.round(data.metrics.D_seats);
    const rseats = Math.round(data.metrics.R_seats);
    document.getElementById("sum-projected").textContent =
      `D ${dseats} / R ${rseats}`;

    const eg = data.metrics.efficiency_gap * 100;
    document.getElementById("m-eg").textContent = fmtSigned(eg) + " (+R / −D)";
    document.getElementById("m-mm").textContent = fmtSigned(data.metrics.mean_median_D * 100);
    document.getElementById("m-pb").textContent = fmtSigned(data.metrics.partisan_bias_D * 100);
  }

  // ---- Hover card ---------------------------------------------------------
  function showHover(e, feat, dist) {
    const p = feat.properties;
    $hover.innerHTML = `
      <div><b>${p.county} County</b></div>
      <div class="row"><span>District</span><span>${p.district_id + 1}</span></div>
      <div class="row"><span>District D / R</span><span>${dist.d_pct.toFixed(1)} / ${dist.r_pct.toFixed(1)}%</span></div>
      <div class="row"><span>County D share</span><span>${(p.d_share * 100).toFixed(1)}%</span></div>
      <div class="row"><span>Population</span><span>${fmtInt(p.population)}</span></div>
      <div class="row"><span>Top group</span><span>${topDemo(p.demographics)}</span></div>
    `;
    $hover.classList.remove("hidden");
    moveHover(e);
  }
  function moveHover(e) {
    const containerRect = document.getElementById("map-wrap").getBoundingClientRect();
    const x = e.originalEvent.clientX - containerRect.left + 14;
    const y = e.originalEvent.clientY - containerRect.top + 14;
    $hover.style.left = Math.min(x, containerRect.width - 250) + "px";
    $hover.style.top = Math.min(y, containerRect.height - 140) + "px";
  }
  function hideHover() { $hover.classList.add("hidden"); }
  function topDemo(demo) {
    const sorted = Object.entries(demo).sort((a, b) => b[1] - a[1]);
    return `${sorted[0][0]} ${(sorted[0][1] * 100).toFixed(0)}%`;
  }

  // ---- Wiring -------------------------------------------------------------
  $state.addEventListener("change", () => {
    syncSlider();
    refresh();
  });
  document.querySelectorAll('input[name="party"]').forEach((el) =>
    el.addEventListener("change", refresh)
  );

  // Debounced sliders.
  let seatsTimer = null;
  $seats.addEventListener("input", () => {
    $seatsDisplay.textContent = $seats.value;
    if (seatsTimer) clearTimeout(seatsTimer);
    seatsTimer = setTimeout(refresh, 200);
  });
  let intensityTimer = null;
  $intensity.addEventListener("input", () => {
    $intensityDisplay.textContent = parseFloat($intensity.value).toFixed(2);
    if (intensityTimer) clearTimeout(intensityTimer);
    intensityTimer = setTimeout(refresh, 200);
  });

  // ---- Boot ---------------------------------------------------------------
  loadStates().then(refresh).catch((err) => {
    console.error(err);
    alert("Failed to initialize: " + err.message);
  });
})();
