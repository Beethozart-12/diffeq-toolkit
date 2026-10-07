// diffeq-toolkit 前端逻辑：切换面板、LaTeX 预览、调用 API、渲染结果。
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

  // ------------------------------------------------------------------ //
  // 面板切换（支持点击与左右方向键，含 ARIA 状态同步）
  // ------------------------------------------------------------------ //
  const tabs = Array.from(document.querySelectorAll(".tab"));
  const panels = { ode: $("panel-ode"), eval: $("panel-eval"), pde: $("panel-pde") };

  function activateTab(name, focus) {
    tabs.forEach((t) => {
      const on = t.dataset.tab === name;
      t.classList.toggle("active", on);
      t.setAttribute("aria-selected", on ? "true" : "false");
      if (on && focus) t.focus();
    });
    Object.keys(panels).forEach((k) => panels[k].classList.toggle("hidden", k !== name));
  }

  tabs.forEach((t, i) => {
    t.addEventListener("click", () => activateTab(t.dataset.tab, false));
    t.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      e.preventDefault();
      const next = (i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length;
      activateTab(tabs[next].dataset.tab, true);
    });
  });

  // ------------------------------------------------------------------ //
  // MathJax 预览
  // ------------------------------------------------------------------ //
  // MathJax 是 async 加载的，可能尚未就绪；轮询至多 10s，就绪后排版当前内容。
  function typeset(el) {
    const attempt = () =>
      !!(window.MathJax && window.MathJax.typesetPromise) &&
      (window.MathJax.typesetPromise([el]).catch(() => {}), true);
    if (attempt()) return;
    const t0 = Date.now();
    const timer = setInterval(() => {
      if (attempt() || Date.now() - t0 > 10000) clearInterval(timer);
    }, 150);
  }

  function refreshPreview(elm) {
    const prev = $(elm.dataset.preview);
    if (!prev) return;
    const txt = elm.value.trim();
    prev.textContent = txt ? "$ " + txt + " $" : "";
    typeset(prev);
  }

  // 输入防抖：连续打字时 120ms 内不重复排版，降低 MathJax 负担。
  function debounce(fn, ms) {
    let timer = null;
    return (...args) => {
      clearTimeout(timer);
      timer = setTimeout(() => fn(...args), ms);
    };
  }

  const odeLatex = $("ode-latex");
  const evalLatex = $("eval-latex");
  odeLatex.dataset.preview = "ode-preview";
  evalLatex.dataset.preview = "eval-preview";
  odeLatex.addEventListener("input", debounce(() => refreshPreview(odeLatex), 120));
  evalLatex.addEventListener("input", debounce(() => refreshPreview(evalLatex), 120));

  // ------------------------------------------------------------------ //
  // 结果渲染（全部走 textContent / DOM API，杜绝注入；NaN 单独显示）
  // ------------------------------------------------------------------ //
  function fmt(v) {
    if (typeof v !== "number") return String(v);
    if (Number.isNaN(v)) return "NaN";
    if (!isFinite(v)) return v > 0 ? "∞" : "-∞";
    return v.toFixed(4);
  }

  function el(tag, cls, text) {
    const d = document.createElement(tag);
    if (cls) d.className = cls;
    if (text !== undefined) d.textContent = text;
    return d;
  }

  // 采样点表格：第一列为行标签（如 t / y），避免表头表体误读。
  function buildSampleTable(xs, ys, xName, yName) {
    const n = Math.min(xs.length, ys.length, 12);
    const table = document.createElement("table");
    const mkRow = (label, vals) => {
      const tr = document.createElement("tr");
      tr.appendChild(el("th", "rowlabel", label));
      for (let i = 0; i < n; i++) tr.appendChild(el("td", "", fmt(vals[i])));
      return tr;
    };
    table.appendChild(mkRow(xName, xs));
    table.appendChild(mkRow(yName, ys));
    return table;
  }

  function renderResult(container, data) {
    container.replaceChildren();
    if (!data.ok) {
      container.appendChild(el("div", "error", "出错了：" + (data.error || "未知错误")));
      return;
    }
    container.appendChild(
      el("div", "python-src", "Python 读得懂的形式：\n" + data.python_src)
    );

    if (data.mode === "ode") {
      container.appendChild(
        el("div", "meta", `自变量 ${data.indep}，因变量 ${data.dep}，共 ${data.n_points} 个解点（下方抽稀显示）。`)
      );
      if (Array.isArray(data.t) && Array.isArray(data.y)) {
        container.appendChild(buildSampleTable(data.t, data.y, data.indep, data.dep));
      }
    } else if (data.mode === "pde") {
      container.appendChild(el("div", "meta", data.scheme + "　|　" + data.meta));
    } else {
      container.appendChild(el("div", "meta", `自变量 ${data.var}。`));
      if (Array.isArray(data.x) && Array.isArray(data.y)) {
        container.appendChild(buildSampleTable(data.x, data.y, data.var, "f(" + data.var + ")"));
      }
    }

    // plot 仅接受 data:image/ 前缀，防止异常 src。
    if (typeof data.plot === "string" && data.plot.startsWith("data:image/")) {
      const img = document.createElement("img");
      img.src = data.plot;
      img.alt = "函数图像";
      container.appendChild(img);
    }
  }

  // ------------------------------------------------------------------ //
  // API 调用（统一入口：加载态 / 错误处理 / 状态提示）
  // ------------------------------------------------------------------ //
  async function postJSON(url, payload) {
    let res;
    try {
      res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    } catch (e) {
      throw new Error("网络连接失败，请确认服务已启动");
    }
    if (!res.ok) throw new Error("服务返回 HTTP " + res.status);
    try {
      return await res.json();
    } catch (e) {
      throw new Error("服务返回了无法解析的数据");
    }
  }

  async function run({ btn, status, result, url, payload, busyText, done }) {
    btn.disabled = true;
    status.textContent = busyText;
    result.replaceChildren();
    try {
      const data = await postJSON(url, payload);
      renderResult(result, data);
      status.textContent = data.ok ? done : "";
    } catch (e) {
      result.replaceChildren(el("div", "error", "请求失败：" + e.message));
      status.textContent = "";
    } finally {
      btn.disabled = false;
    }
  }

  const odeRun = $("ode-run");
  const odeStatus = $("ode-status");
  const odeResult = $("ode-result");

  function submitOde() {
    run({
      btn: odeRun,
      status: odeStatus,
      result: odeResult,
      url: "/api/solve",
      payload: {
        latex: odeLatex.value,
        y0: $("ode-y0").value,
        t0: $("ode-t0").value,
        tf: $("ode-tf").value,
      },
      busyText: "求解中…",
      done: "完成",
    });
  }
  odeRun.addEventListener("click", submitOde);

  const evalRun = $("eval-run");
  const evalStatus = $("eval-status");
  const evalResult = $("eval-result");

  function submitEval() {
    run({
      btn: evalRun,
      status: evalStatus,
      result: evalResult,
      url: "/api/eval",
      payload: {
        latex: evalLatex.value,
        var: $("eval-var").value,
        a: $("eval-a").value,
        b: $("eval-b").value,
      },
      busyText: "计算中…",
      done: "完成",
    });
  }
  evalRun.addEventListener("click", submitEval);

  // ------------------------------------------------------------------ //
  // PDE 面板：类型/维数联动 + 求解
  // ------------------------------------------------------------------ //
  const pdeType = $("pde-type");
  const pdeDims = $("pde-dims");
  const pdeEq = $("pde-eq");
  const pdeLatex = $("pde-latex");
  const pdeBc = $("pde-bc");
  const pdeIcLabel = $("pde-ic-label");
  const pdeCoeffLabel = $("pde-coeff-label");
  const pdePreview = $("pde-preview");

  // 各类型/维数下的初值占位（切到 laplace 时是「初始猜测」）
  const PDE_PLACEHOLDERS = {
    heat: ["\\sin(\\pi x)", "\\sin(\\pi x)\\sin(\\pi y)",
           "\\sin(\\pi x)\\sin(\\pi y)\\sin(\\pi z)"],
    wave: ["\\sin(\\pi x)", "\\sin(\\pi x)\\sin(\\pi y)",
           "\\sin(\\pi x)\\sin(\\pi y)\\sin(\\pi z)"],
    laplace: ["0", "0", "0"],
  };
  const PDE_DIM_NAMES = ["x", "x, y", "x, y, z"];

  function syncPdeForm() {
    const type = pdeType.value;
    const dims = Number(pdeDims.value);
    // Laplace 至少 2 维：禁用 1 维选项，必要时自动切到 2 维
    pdeDims.querySelector('option[value="1"]').disabled = type === "laplace";
    if (type === "laplace" && dims === 1) {
      pdeDims.value = "2";
    }
    const d = Number(pdeDims.value);
    const isLaplace = type === "laplace";
    document.querySelectorAll(".pde-heatwave").forEach((n) => n.classList.toggle("hidden", isLaplace));
    document.querySelectorAll(".pde-laplace").forEach((n) => n.classList.toggle("hidden", !isLaplace));
    pdeIcLabel.textContent = (isLaplace ? "初始猜测（LaTeX，关于 " : "初值（LaTeX，关于 ")
      + PDE_DIM_NAMES[d - 1] + "）";
    pdeCoeffLabel.textContent = type === "wave" ? "c 波速" : "α 热扩散系数";
    pdeLatex.placeholder = PDE_PLACEHOLDERS[type][d - 1];
    refreshPreview(pdeLatex);
  }
  pdeType.addEventListener("change", syncPdeForm);
  pdeDims.addEventListener("change", syncPdeForm);

  const pdeRun = $("pde-run");
  const pdeStatus = $("pde-status");
  const pdeResult = $("pde-result");

  function submitPde() {
    run({
      btn: pdeRun,
      status: pdeStatus,
      result: pdeResult,
      url: "/api/pde",
      payload: {
        eq: pdeEq.value,
        type: pdeType.value,
        dims: Number(pdeDims.value),
        latex: pdeLatex.value,
        bc: pdeBc.value,
        coeff: $("pde-coeff").value,
        L: $("pde-L").value,
        n: $("pde-n").value,
        dt: $("pde-dt").value,
        steps: $("pde-steps").value,
        tol: $("pde-tol").value,
        maxiter: $("pde-maxiter").value,
      },
      busyText: "求解中（PDE 计算量较大，请稍候）…",
      done: "完成",
    });
  }
  pdeRun.addEventListener("click", submitPde);

  // Ctrl/Cmd + Enter 快捷提交
  const submitByIndex = [submitOde, submitEval, submitPde];
  [odeLatex, evalLatex, pdeLatex].forEach((ta, idx) => {
    ta.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        submitByIndex[idx]();
      }
    });
  });

  // 初始化预览（MathJax 未就绪时由 typeset 内部轮询补齐）
  pdeEq.dataset.preview = "pde-eq-preview";
  pdeEq.addEventListener("input", debounce(() => refreshPreview(pdeEq), 120));
  pdeLatex.dataset.preview = "pde-preview";
  syncPdeForm();
  refreshPreview(odeLatex);
  refreshPreview(evalLatex);
  refreshPreview(pdeEq);
})();
