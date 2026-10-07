// diffeq-toolkit 前端逻辑：切换面板、LaTeX 预览、调用 API、渲染结果。
(function () {
  "use strict";

  // ---- 面板切换 ----
  const tabs = document.querySelectorAll(".tab");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      const name = tab.dataset.tab;
      document.getElementById("panel-ode").classList.toggle("hidden", name !== "ode");
      document.getElementById("panel-eval").classList.toggle("hidden", name !== "eval");
    });
  });

  // ---- MathJax 预览 ----
  function refreshPreview(elm) {
    const txt = elm.value.trim();
    const prev = document.getElementById(elm.dataset.preview);
    prev.textContent = txt ? "$ " + txt + " $" : "";
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise([prev]).catch(() => {});
    }
  }
  const odeLatex = document.getElementById("ode-latex");
  const evalLatex = document.getElementById("eval-latex");
  odeLatex.dataset.preview = "ode-preview";
  evalLatex.dataset.preview = "eval-preview";
  odeLatex.addEventListener("input", () => refreshPreview(odeLatex));
  evalLatex.addEventListener("input", () => refreshPreview(evalLatex));

  // ---- 渲染结果 ----
  function renderResult(container, data) {
    container.innerHTML = "";
    if (!data.ok) {
      const d = document.createElement("div");
      d.className = "error";
      d.textContent = "出错了：" + data.error;
      container.appendChild(d);
      return;
    }
    const src = document.createElement("div");
    src.className = "python-src";
    src.textContent = "Python 读得懂的形式：\n" + data.python_src;
    container.appendChild(src);

    const meta = document.createElement("div");
    meta.className = "meta";
    if (data.mode === "ode") {
      meta.textContent = `自变量 ${data.indep}，因变量 ${data.dep}，共 ${data.n_points} 个解点（下方抽稀显示）。`;
    } else {
      meta.textContent = `自变量 ${data.var}。`;
    }
    container.appendChild(meta);

    if (Array.isArray(data.x) && Array.isArray(data.y)) {
      const n = Math.min(data.x.length, 12);
      const table = document.createElement("table");
      let head = "<tr>";
      for (let i = 0; i < n; i++) head += "<th>" + fmt(data.x[i]) + "</th>";
      head += "</tr>";
      let body = "<tr>";
      for (let i = 0; i < n; i++) body += "<td>" + fmt(data.y[i]) + "</td>";
      body += "</tr>";
      table.innerHTML = head + body;
      container.appendChild(table);
    }

    if (data.plot) {
      const img = document.createElement("img");
      img.src = data.plot;
      container.appendChild(img);
    }
  }

  function fmt(v) {
    if (typeof v !== "number") return v;
    if (!isFinite(v)) return v > 0 ? "∞" : "-∞";
    return v.toFixed(4);
  }

  // ---- 调用 API ----
  async function postJSON(url, payload) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return res.json();
  }

  const odeRun = document.getElementById("ode-run");
  odeRun.addEventListener("click", async () => {
    odeRun.disabled = true;
    try {
      const data = await postJSON("/api/solve", {
        latex: odeLatex.value,
        y0: document.getElementById("ode-y0").value,
        t0: document.getElementById("ode-t0").value,
        tf: document.getElementById("ode-tf").value,
      });
      renderResult(document.getElementById("ode-result"), data);
    } catch (e) {
      document.getElementById("ode-result").innerHTML =
        '<div class="error">请求失败：' + e + "</div>";
    } finally {
      odeRun.disabled = false;
    }
  });

  const evalRun = document.getElementById("eval-run");
  evalRun.addEventListener("click", async () => {
    evalRun.disabled = true;
    try {
      const data = await postJSON("/api/eval", {
        latex: evalLatex.value,
        var: document.getElementById("eval-var").value,
        a: document.getElementById("eval-a").value,
        b: document.getElementById("eval-b").value,
      });
      renderResult(document.getElementById("eval-result"), data);
    } catch (e) {
      document.getElementById("eval-result").innerHTML =
        '<div class="error">请求失败：' + e + "</div>";
    } finally {
      evalRun.disabled = false;
    }
  });

  // 初始化预览
  refreshPreview(odeLatex);
  refreshPreview(evalLatex);
})();
