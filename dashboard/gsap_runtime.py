"""
Inject [GSAP](https://gsap.com/) into the Streamlit parent page.

Streamlit's ``components.html`` runs inside a sandboxed iframe, so animations
must be attached to ``parent.document``. The runtime loads GSAP from jsDelivr
once, then re-plays entrance tweens whenever Streamlit rerenders the DOM.
"""

from __future__ import annotations

import streamlit as st

GSAP_CDN = "https://cdn.jsdelivr.net/npm/gsap@3.13.0/dist/gsap.min.js"
SCROLL_CDN = "https://cdn.jsdelivr.net/npm/gsap@3.13.0/dist/ScrollTrigger.min.js"

_GSAP_BOOTSTRAP = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <style>html,body{{margin:0;height:0;overflow:hidden;background:transparent;}}</style>
</head>
<body>
<script>
(function () {{
  function rootWindow() {{
    var w = window;
    for (var i = 0; i < 8; i++) {{
      try {{
        if (!w.parent || w.parent === w) break;
        void w.parent.document.body;
        w = w.parent;
      }} catch (err) {{
        break;
      }}
    }}
    return w;
  }}

  var win = rootWindow();
  var doc = win.document;

  function loadScript(src) {{
    return new Promise(function (resolve, reject) {{
      var id = "mdb-" + src.split("/").pop().replace(/\\W/g, "");
      if (src.indexOf("gsap.min.js") !== -1 && win.gsap) {{
        resolve();
        return;
      }}
      if (src.indexOf("ScrollTrigger") !== -1 && win.ScrollTrigger) {{
        resolve();
        return;
      }}
      var existing = doc.getElementById(id);
      if (existing) {{
        if (existing.getAttribute("data-ok") === "1") resolve();
        else existing.addEventListener("load", function () {{ resolve(); }});
        return;
      }}
      var s = doc.createElement("script");
      s.id = id;
      s.src = src;
      s.async = true;
      s.onload = function () {{ s.setAttribute("data-ok", "1"); resolve(); }};
      s.onerror = reject;
      doc.head.appendChild(s);
    }});
  }}

  function bindHover(gsap, selector, enterVars, leaveVars) {{
    doc.querySelectorAll(selector).forEach(function (el) {{
      if (el.dataset.gsapHover === "1") return;
      el.dataset.gsapHover = "1";
      el.addEventListener("mouseenter", function () {{ gsap.to(el, enterVars); }});
      el.addEventListener("mouseleave", function () {{ gsap.to(el, leaveVars); }});
    }});
  }}

  function fresh(sel) {{
    return Array.prototype.slice.call(doc.querySelectorAll(sel)).filter(function (el) {{
      return el.dataset.gsapPlayed !== "1";
    }});
  }}

  function mark(els) {{
    els.forEach(function (el) {{ el.dataset.gsapPlayed = "1"; }});
  }}

  function splitHeading(h1) {{
    if (!h1 || h1.dataset.split === "1") return;
    var text = h1.textContent || "";
    h1.innerHTML = text.split("").map(function (ch) {{
      return '<span class="gsap-ch">' + (ch === " " ? "&nbsp;" : ch) + "</span>";
    }}).join("");
    h1.dataset.split = "1";
  }}

  function animateFresh(gsap) {{
    var hero = fresh(".hero");
    if (hero.length) {{
      gsap.from(hero, {{ opacity: 0, y: 32, duration: 0.7, ease: "power3.out" }});
      hero.forEach(function (node) {{
        var h1 = node.querySelector("h1");
        splitHeading(h1);
        if (h1) {{
          gsap.from(h1.querySelectorAll(".gsap-ch"), {{
            y: 16,
            opacity: 0,
            rotateX: -50,
            stagger: 0.016,
            duration: 0.48,
            ease: "back.out(1.5)",
            delay: 0.12
          }});
        }}
        var kicker = node.querySelector(".hero-kicker");
        var para = node.querySelector("p");
        if (kicker) gsap.from(kicker, {{ opacity: 0, y: 8, duration: 0.4, delay: 0.05 }});
        if (para) gsap.from(para, {{ opacity: 0, y: 10, duration: 0.5, delay: 0.22 }});
      }});
      mark(hero);
    }}

    var kpis = fresh(".kpi-card");
    if (kpis.length) {{
      gsap.from(kpis, {{
        opacity: 0, y: 26, scale: 0.96, stagger: 0.08, duration: 0.55, ease: "power2.out"
      }});
      mark(kpis);
    }}

    var pipes = fresh(".pipe-step");
    if (pipes.length) {{
      gsap.from(pipes, {{
        opacity: 0, scale: 0.82, stagger: 0.07, duration: 0.42, ease: "back.out(1.7)"
      }});
      mark(pipes);
    }}

    var arrows = fresh(".pipe-arrow");
    if (arrows.length) {{
      gsap.to(arrows, {{
        x: 7, opacity: 0.4, duration: 0.85, yoyo: true, repeat: -1, ease: "sine.inOut", stagger: 0.1
      }});
      mark(arrows);
    }}

    var tech = fresh(".tech-card");
    if (tech.length) {{
      gsap.from(tech, {{ opacity: 0, y: 18, stagger: 0.05, duration: 0.42, ease: "power2.out" }});
      mark(tech);
    }}

    var titles = fresh(".section-title");
    if (titles.length) {{
      gsap.from(titles, {{ opacity: 0, y: 12, stagger: 0.05, duration: 0.4, ease: "power2.out" }});
      mark(titles);
    }}

    var charts = fresh('div[data-testid="stPlotlyChart"]');
    if (charts.length) {{
      gsap.from(charts, {{ opacity: 0, y: 22, stagger: 0.07, duration: 0.55, ease: "power2.out" }});
      mark(charts);
    }}

    var frames = fresh('div[data-testid="stDataFrame"]');
    if (frames.length) {{
      gsap.from(frames, {{ opacity: 0, y: 14, duration: 0.45, ease: "power2.out" }});
      mark(frames);
    }}

    var result = fresh(".result-card");
    if (result.length) {{
      gsap.from(result, {{ opacity: 0, scale: 0.92, duration: 0.5, ease: "back.out(1.6)" }});
      mark(result);
    }}

    var fills = fresh(".score-fill");
    if (fills.length) {{
      fills.forEach(function (el) {{
        var w = el.style.width || "0%";
        gsap.fromTo(el, {{ width: "0%" }}, {{ width: w, duration: 0.9, ease: "power2.out" }});
      }});
      mark(fills);
    }}

    var brand = fresh(".brand-title");
    if (brand.length) {{
      gsap.from(brand, {{ opacity: 0, x: -14, duration: 0.5, ease: "power3.out" }});
      mark(brand);
    }}

    var pill = fresh(".status-pill");
    if (pill.length) {{
      gsap.from(pill, {{ opacity: 0, y: 6, duration: 0.4, delay: 0.15 }});
      mark(pill);
    }}

    var nav = fresh('section[data-testid="stSidebar"] label[data-baseweb="radio"]');
    if (nav.length) {{
      gsap.from(nav, {{ opacity: 0, x: -12, stagger: 0.035, duration: 0.32, ease: "power2.out" }});
      mark(nav);
    }}

    var empty = fresh(".empty-note");
    if (empty.length) {{
      gsap.from(empty, {{ opacity: 0, y: 10, duration: 0.4 }});
      mark(empty);
    }}

    bindHover(
      gsap,
      ".kpi-card",
      {{ y: -8, duration: 0.24, ease: "power2.out", overwrite: "auto" }},
      {{ y: 0, duration: 0.24, ease: "power2.out", overwrite: "auto" }}
    );
    bindHover(
      gsap,
      ".tech-card",
      {{ y: -5, duration: 0.2, ease: "power2.out", overwrite: "auto" }},
      {{ y: 0, duration: 0.2, ease: "power2.out", overwrite: "auto" }}
    );
    bindHover(
      gsap,
      ".pipe-step",
      {{ scale: 1.07, duration: 0.18, ease: "power2.out", overwrite: "auto" }},
      {{ scale: 1, duration: 0.18, ease: "power2.out", overwrite: "auto" }}
    );
  }}

  async function boot() {{
    try {{
      await loadScript("{GSAP_CDN}");
      try {{ await loadScript("{SCROLL_CDN}"); }} catch (err) {{}}
    }} catch (err) {{
      return;
    }}
    var gsap = win.gsap;
    if (!gsap) return;
    if (win.ScrollTrigger && gsap.registerPlugin) {{
      gsap.registerPlugin(win.ScrollTrigger);
    }}

    var run = function () {{ animateFresh(gsap); }};
    run();
    if (!win.__mdbGsapObserver) {{
      var timer;
      win.__mdbGsapObserver = new MutationObserver(function () {{
        clearTimeout(timer);
        timer = setTimeout(run, 140);
      }});
      win.__mdbGsapObserver.observe(doc.body, {{ childList: true, subtree: true }});
    }}
  }}

  boot();
}})();
</script>
</body>
</html>
"""


def inject_gsap() -> None:
    """Load GSAP in the parent document and animate dashboard UI elements."""
    st.iframe(_GSAP_BOOTSTRAP, height=1, width=1)
