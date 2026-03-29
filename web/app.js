const state = {
  step: 0,
  domains: [],
  selectedDomains: new Set(),
  customDomainOpen: false,
  perGroupTextMode: false,
};

const $ = (id) => document.getElementById(id);

let _toastTimer = null;

function showToast(msg, type = "error") {
  const el = $("toast");
  if (!el) return;
  if (_toastTimer) clearTimeout(_toastTimer);
  el.textContent = msg;
  el.className = `toast show ${type}`;
  _toastTimer = setTimeout(() => {
    el.classList.remove("show");
  }, 4500);
}

function escapeHtml(text) {
  return String(text ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

const CHAT_MESSAGES = [
  "<p><strong>Bonjour.</strong> Je suis RecruitAI, votre assistant de staffing associatif.</p><p>Question 1 : Quels domaines de votre evenement souhaitez-vous couvrir ?</p>",
  "<p>Parfait ! Combien de candidats souhaitez-vous par groupe de travail ?</p>",
  "<p>Analyse complete. Voici les groupes formes selon les profils de vos candidats.</p>",
];

function updateChat(step) {
  const body = $("chat-body");
  if (body && CHAT_MESSAGES[step]) {
    body.innerHTML = CHAT_MESSAGES[step];
  }
}

function setStep(step) {
  state.step = step;
  [$("step-0"), $("step-1"), $("step-2")].forEach((el, idx) => {
    el.classList.toggle("hidden", idx !== step);
  });

  const stepNodes = Array.from(document.querySelectorAll(".step"));
  const lineNodes = Array.from(document.querySelectorAll(".line"));

  stepNodes.forEach((el, idx) => {
    el.classList.remove("active", "done");
    if (idx < step) {
      el.classList.add("done");
      el.textContent = "✓";
    } else if (idx === step) {
      el.classList.add("active");
      el.textContent = String(idx + 1);
    } else {
      el.textContent = String(idx + 1);
    }
  });

  lineNodes.forEach((line, idx) => {
    line.classList.toggle("done", idx < step);
  });

  updateChat(step);
}

function renderDomainButtons() {
  const grid = $("domain-grid");
  grid.innerHTML = "";

  state.domains.forEach((domain) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "domain-btn";
    btn.style.setProperty("--c", domain.color);

    const isActive = state.selectedDomains.has(domain.label);
    if (isActive) btn.classList.add("active");

    btn.innerHTML = `<span class="domain-icon">${escapeHtml(domain.icon)}</span><span>${escapeHtml(domain.label)}</span>`;

    btn.addEventListener("click", () => {
      if (state.selectedDomains.has(domain.label)) {
        state.selectedDomains.delete(domain.label);
      } else {
        state.selectedDomains.add(domain.label);
      }
      renderDomainButtons();
      renderSelectedDomains();
    });

    grid.appendChild(btn);
  });
}

function renderSelectedDomains() {
  const container = $("selected-domains");
  const names = Array.from(state.selectedDomains);
  container.innerHTML = names
    .map((name) => `<span class="chip">${escapeHtml(name)}</span>`)
    .join("");
}

async function loadDomains() {
  const res = await fetch("/api/domains");
  if (!res.ok) throw new Error("Impossible de charger les domaines");
  const payload = await res.json();
  state.domains = payload.domains || [];
  renderDomainButtons();
}

function formatScore(score) {
  const val = parseFloat(score) || 0;
  if (val <= 0) return '<span class="score-badge score-none">—</span>';
  if (val < 1) {
    const pct = Math.round(val * 100);
    const cls = pct >= 70 ? "score-high" : pct >= 40 ? "score-med" : "score-low";
    return `<span class="score-badge ${cls}">${pct}%</span>`;
  }
  const cls = val >= 5 ? "score-high" : val >= 2 ? "score-med" : "score-low";
  return `<span class="score-badge ${cls}">${val.toFixed(0)} pts</span>`;
}

function renderStats(stats) {
  $("stats").innerHTML = `
    <div class="stat"><div class="v">${escapeHtml(stats.total_candidates)}</div><div class="l">Candidats analyses</div></div>
    <div class="stat"><div class="v">${escapeHtml(stats.group_count)}</div><div class="l">Groupes formes</div></div>
    <div class="stat"><div class="v">${escapeHtml(stats.max_per_group)}</div><div class="l">Max par groupe</div></div>
  `;
}

function renderErrors(errors) {
  const box = $("api-errors");
  if (!errors || !errors.length) { box.innerHTML = ""; return; }
  const lines = errors.map((e) => `<li>${escapeHtml(e)}</li>`).join("");
  box.innerHTML = `<div class="error-box"><strong>Fichiers non traites :</strong><ul>${lines}</ul></div>`;
}

function buildExplanation(candidate, domain) {
  const sentences = [];
  const name = escapeHtml(candidate.name || "Ce candidat");
  const dom = escapeHtml(domain);

  // Core reason: why this domain
  const kw = candidate.matched_kw && candidate.matched_kw.length > 0
    ? candidate.matched_kw
    : null;
  const skills = candidate.skills && candidate.skills.length > 0
    ? candidate.skills.slice(0, 3)
    : null;

  if (kw) {
    sentences.push(
      `${name} a été placé(e) dans le groupe <strong>${dom}</strong> car son CV reflète directement les besoins de ce domaine, notamment à travers les termes : <em>${escapeHtml(kw.join(", "))}</em>.`
    );
  } else if (skills) {
    sentences.push(
      `${name} a été placé(e) dans le groupe <strong>${dom}</strong> grâce à ses compétences en <em>${escapeHtml(skills.join(", "))}</em>, qui correspondent au profil recherché pour ce domaine.`
    );
  } else {
    sentences.push(
      `${name} a été placé(e) dans le groupe <strong>${dom}</strong> par correspondance sémantique avec les profils attendus dans ce domaine.`
    );
  }

  // Skills complement (only if keywords already opened)
  if (kw && skills) {
    sentences.push(
      `Ses compétences en <em>${escapeHtml(skills.join(", "))}</em> renforcent sa pertinence pour ce groupe.`
    );
  }

  // Education
  if (candidate.education && candidate.education !== "Not specified") {
    sentences.push(
      `Sa formation en <em>${escapeHtml(candidate.education)}</em> constitue un atout supplémentaire pour contribuer efficacement aux activités du domaine.`
    );
  }

  // Summary (AI-generated profile, skip auto-extraction noise)
  if (candidate.summary) {
    sentences.push(escapeHtml(candidate.summary));
  }

  // Experience
  const exp = parseInt(candidate.experience, 10);
  if (exp > 0) {
    sentences.push(
      `Avec ${exp} an${exp > 1 ? "s" : ""} d'expérience, ce profil apporte une maturité appréciable au sein du groupe.`
    );
  }

  if (sentences.length === 0) return "";

  return `<p class="expl-text">${sentences.join(" ")}</p>`;
}

function renderGroups(groups) {
  const root = $("groups");
  if (!groups || !groups.length) {
    root.innerHTML = "<p style='color:#94a3b8;text-align:center;padding:1rem'>Aucun CV trouve dans le dossier cvs.</p>";
    return;
  }

  root.innerHTML = groups
    .map((group) => {
      const border = `${group.color}66`;
      const soft = `${group.color}22`;
      const isUnclassified = /non clas/i.test(group.domain);

      const candidates = (group.candidates || [])
        .map((candidate) => {
          const skills = (candidate.skills || [])
            .map((s) => `<span class="tag">${escapeHtml(s)}</span>`)
            .join("");
          const explanation = buildExplanation(candidate, group.domain);

          return `
            <div class="candidate">
              <div class="candidate-header" role="button" tabindex="0" aria-expanded="false">
                <div class="candidate-header-left">
                  <div class="candidate-name">
                    ${escapeHtml(candidate.name || "Inconnu")}
                    <svg class="chevron" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M4 6l4 4 4-4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                  </div>
                  <div class="candidate-meta">${escapeHtml(candidate.department || "General")}</div>
                </div>
                ${formatScore(candidate.score)}
              </div>
              <div class="expl-panel">
                ${explanation}
                ${skills ? `<div class="tag-row" style="padding:0 0 0.4rem">${skills}</div>` : ""}
              </div>
            </div>
          `;
        })
        .join("");

      return `
        <article class="group" style="--c-border:${border};--c-soft:${soft}">
          <div class="group-head">
            <div class="group-title">${escapeHtml(group.icon)} ${escapeHtml(group.domain)}</div>
            <div class="group-count">${escapeHtml(group.count)} profil${group.count > 1 ? "s" : ""}</div>
          </div>
          ${isUnclassified ? `<div class="group-summary">${escapeHtml(group.summary || "")}</div>` : ""}
          ${candidates}
        </article>
      `;
    })
    .join("");

  // Bind click-to-expand on every candidate header
  root.querySelectorAll(".candidate-header").forEach((header) => {
    const toggle = () => {
      const panel = header.nextElementSibling;
      const isOpen = header.getAttribute("aria-expanded") === "true";
      header.setAttribute("aria-expanded", String(!isOpen));
      panel.classList.toggle("open", !isOpen);
    };
    header.addEventListener("click", toggle);
    header.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); }
    });
  });
}

async function analyze() {
  const loading = $("loading");
  const analyzeBtn = $("analyze");

  setStep(2);
  $("stats").innerHTML = "";
  $("groups").innerHTML = "";
  $("api-errors").innerHTML = "";
  loading.classList.remove("hidden");
  analyzeBtn.disabled = true;

  const customDomain = state.customDomainOpen ? $("custom-domain").value.trim() : "";
  const perGroup = Number($("per-group").value || 3);
  const perGroupText = state.perGroupTextMode ? $("per-group-text").value.trim() : "";

  const body = {
    domains: Array.from(state.selectedDomains),
    custom_domain: customDomain || null,
    per_group: perGroup,
    per_group_text: perGroupText || null,
  };

  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), 120000);

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    const payload = await res.json();
    if (!res.ok) throw new Error(payload.detail || "Analyse impossible");

    renderStats(payload.stats || {});
    renderErrors(payload.errors || []);
    renderGroups(payload.groups || []);
  } catch (err) {
    if (err.name === "AbortError") {
      showToast("Analyse trop longue (timeout). Verifiez le dossier cvs puis reessayez.", "error");
    } else {
      showToast(err.message || "Erreur inattendue", "error");
    }
    setStep(1);
  } finally {
    window.clearTimeout(timeoutId);
    loading.classList.add("hidden");
    analyzeBtn.disabled = false;
  }
}

function restartFlow() {
  state.step = 0;
  state.selectedDomains = new Set();
  state.customDomainOpen = false;
  state.perGroupTextMode = false;

  $("custom-domain").value = "";
  $("per-group").value = "3";
  $("per-group-text").value = "";

  $("custom-domain-wrap").classList.add("hidden");
  $("per-group-text-wrap").classList.add("hidden");

  renderDomainButtons();
  renderSelectedDomains();
  $("stats").innerHTML = "";
  $("groups").innerHTML = "";
  $("api-errors").innerHTML = "";

  setStep(0);
}

function bindUi() {
  $("toggle-custom-domain").addEventListener("click", () => {
    state.customDomainOpen = !state.customDomainOpen;
    $("custom-domain-wrap").classList.toggle("hidden", !state.customDomainOpen);
    $("toggle-custom-domain").textContent = state.customDomainOpen ? "— Annuler" : "+ Autre domaine";
  });

  $("toggle-per-group-text").addEventListener("click", () => {
    state.perGroupTextMode = !state.perGroupTextMode;
    $("per-group-text-wrap").classList.toggle("hidden", !state.perGroupTextMode);
    $("toggle-per-group-text").textContent = state.perGroupTextMode ? "Annuler" : "Texte libre";
  });

  $("to-step-1").addEventListener("click", () => {
    const custom = state.customDomainOpen ? $("custom-domain").value.trim() : "";
    if (state.selectedDomains.size === 0 && !custom) {
      showToast("Selectionnez au moins un domaine", "warn");
      return;
    }
    setStep(1);
  });

  $("back-step-0").addEventListener("click", () => setStep(0));
  $("analyze").addEventListener("click", analyze);
  $("restart").addEventListener("click", restartFlow);
}

function startBackgroundAnimation() {
  const canvas = $("bg-canvas");
  if (!canvas) return;

  const ctx = canvas.getContext("2d", { alpha: true });
  let w = 0, h = 0, pts = [];
  const colors = ["#64DCFF", "#7DD3FC", "#67E8F9", "#A5B4FC"];
  const mouse = { x: -9999, y: -9999 };

  function rand(a, b) { return a + Math.random() * (b - a); }
  function rgb(hex) {
    return `${parseInt(hex.slice(1,3),16)},${parseInt(hex.slice(3,5),16)},${parseInt(hex.slice(5,7),16)}`;
  }

  function resize() {
    w = canvas.width = window.innerWidth;
    h = canvas.height = window.innerHeight;
    const n = w < 760 ? 50 : 100;
    pts = [];
    for (let i = 0; i < n; i++) {
      pts.push({ x: rand(0,w), y: rand(0,h), vx: rand(-0.3,0.3), vy: rand(-0.3,0.3),
        r: rand(0.5,2.0), c: colors[Math.floor(Math.random()*colors.length)], o: rand(0.45,0.85) });
    }
  }

  function draw() {
    ctx.clearRect(0, 0, w, h);
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      for (let j = i+1; j < pts.length; j++) {
        const q = pts[j];
        const d = Math.hypot(p.x-q.x, p.y-q.y);
        if (d < 140) {
          ctx.beginPath(); ctx.moveTo(p.x,p.y); ctx.lineTo(q.x,q.y);
          ctx.strokeStyle = `rgba(0,200,255,${(1-d/140)*0.24})`; ctx.lineWidth=1; ctx.stroke();
        }
      }
      const md = Math.hypot(p.x-mouse.x, p.y-mouse.y);
      if (md < 170 && md > 0) {
        ctx.beginPath(); ctx.moveTo(p.x,p.y); ctx.lineTo(mouse.x,mouse.y);
        ctx.strokeStyle = `rgba(100,220,255,${(1-md/170)*0.26})`; ctx.lineWidth=1; ctx.stroke();
      }
      ctx.shadowBlur=8; ctx.shadowColor=p.c;
      ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
      ctx.fillStyle = `rgba(${rgb(p.c)},${p.o})`; ctx.fill();
      ctx.shadowBlur=0;
      p.vx += (Math.random()-0.5)*0.02; p.vy += (Math.random()-0.5)*0.02;
      p.vx *= 0.984; p.vy *= 0.984;
      const speed = Math.hypot(p.vx,p.vy);
      if (speed>1.6) { p.vx=(p.vx/speed)*1.6; p.vy=(p.vy/speed)*1.6; }
      p.x+=p.vx; p.y+=p.vy;
      if (p.x<-10) p.x=w+10; if (p.x>w+10) p.x=-10;
      if (p.y<-10) p.y=h+10; if (p.y>h+10) p.y=-10;
    }
    window.requestAnimationFrame(draw);
  }

  window.addEventListener("resize", resize);
  window.addEventListener("mousemove", (e) => { mouse.x=e.clientX; mouse.y=e.clientY; });
  window.addEventListener("mouseleave", () => { mouse.x=-9999; mouse.y=-9999; });
  resize(); draw();
}

async function init() {
  bindUi();
  setStep(0);
  startBackgroundAnimation();
  try {
    await loadDomains();
  } catch (err) {
    showToast(err.message || "Erreur chargement domaines", "error");
  }
}

window.addEventListener("DOMContentLoaded", init);
