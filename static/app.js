// CineMatch – front-end logic: calls the app.py APIs and renders the page.
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const state = { info: null, profiles: [], byId: new Map(), user: null, n: 10, measure: "cosine" };

  const MEASURE_NAMES = { cosine: "Cosine", pearson: "Pearson", jaccard: "Jaccard" };
  const QUICK_COUNT = 8;

  const fmt = new Intl.NumberFormat("en-US");
  const dec = (x, d = 1) => Number(x).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  const genreList = (gs) => gs;
  // Fallback messages in case the server response has no error text.
  const ERRORS = {
    400: "Invalid request: check the profile ID, the number of movies (2 to 20) and the measure.",
    404: "No profile with this ID.",
  };

  async function api(path) {
    const res = await fetch(path);
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.error || ERRORS[res.status] || `The server returned error ${res.status}.`);
    return body;
  }

  // Build DOM nodes (no HTML string concatenation, so data cannot inject markup)
  function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") node.className = v;
      else if (k === "style") node.style.cssText = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else if (v !== null && v !== undefined) node.setAttribute(k, v);
    }
    for (const c of children.flat()) if (c !== null && c !== undefined && c !== false) node.append(c);
    return node;
  }
  // Text with **bold** -> nodes
  function rich(text) {
    return text.split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((t) => (t.startsWith("**") ? el("strong", {}, t.slice(2, -2)) : t));
  }
  function setStatus(id, text, isError = false) {
    $(id).textContent = text;
    $(id).classList.toggle("is-error", isError);
  }

  // ================= DEMO =================
  function renderQuick() {
    $("quick-list").replaceChildren(...state.profiles.slice(0, QUICK_COUNT).map((p) => el("li", {},
      el("button", {
        type: "button", "data-user": p.userId, "aria-current": String(p.userId === state.user),
        title: `${fmt.format(p.n_ratings)} movies rated`, onclick: () => chooseUser(p.userId),
      }, `#${p.userId}`))));
  }

  function renderOptions() {
    $("profile-options").replaceChildren(...state.profiles.map((p) =>
      el("option", { value: String(p.userId) }, `${fmt.format(p.n_ratings)} movies, ${genreList(p.top_genres).slice(0, 2).join(", ")}`)));
  }

  function submitProfile() {
    const raw = $("profile-input").value.trim();
    const id = Number(raw);
    if (!raw || !state.byId.has(id)) {
      setStatus("rec-status", `There is no profile with ID ${raw || "(empty)"}. Profile IDs range from 1 to ${state.info.n_users}.`, true);
      return;
    }
    chooseUser(id);
  }

  function movieRow(rank, m, value, extra) {
    return el("li", { class: "movie" },
      el("span", { class: "movie__rank", "aria-label": `Rank ${rank}` }, String(rank)),
      el("h4", { class: "movie__title" }, m.title, m.year ? el("span", { class: "movie__year" }, ` (${m.year})`) : null),
      el("div", { class: "movie__score" },
        el("strong", {}, dec(value)), el("small", {}, " / 5"),
        el("div", { class: "movie__bar", "aria-hidden": "true" }, el("span", { style: `width:${(value / 5) * 100}%` }))),
      el("p", { class: "movie__meta" }, genreList(m.genres).slice(0, 3).join(", "), extra));
  }
  const skeletons = (n) => Array.from({ length: n }, () => el("li", { class: "skeleton", "aria-hidden": "true" }));

  async function loadRecommendations() {
    if (state.user === null) return;
    const list = $("rec-list");
    list.replaceChildren(...skeletons(Math.min(state.n, 5)));
    setStatus("rec-status", "Computing recommendations…");
    try {
      const data = await api(`/api/recommend?user=${state.user}&n=${state.n}&measure=${state.measure}`);
      if (!data.items.length) {
        list.replaceChildren();
        setStatus("rec-status", "This profile has too few similar movies for recommendations. Try another measure.");
        return;
      }
      list.replaceChildren(...data.items.map((m, i) => movieRow(i + 1, m, m.predicted,
        [". Because this profile liked ", el("b", {}, m.because.title)])));
      setStatus("rec-status", `${data.items.length} unseen movies, ranked by similarity to the profile. The right column is the predicted rating. ${MEASURE_NAMES[data.measure]} measure, ${dec(data.elapsed_ms)} ms.`);
    } catch (err) {
      list.replaceChildren();
      setStatus("rec-status", err.message, true);
    }
  }

  async function loadWatched() {
    if (state.user === null) return;
    try {
      const data = await api(`/api/watched?user=${state.user}&n=10`);
      $("watched-list").replaceChildren(...data.items.map((m, i) => movieRow(i + 1, m, m.rating)));
    } catch (err) {
      $("watched-list").replaceChildren(el("li", { class: "status is-error" }, err.message));
    }
  }

  function chooseUser(userId) {
    state.user = userId;
    const p = state.byId.get(userId);
    $("profile-input").value = String(userId);
    $("profile-title").textContent = `Profile #${userId}`;
    $("profile-stats").textContent = p ? `${fmt.format(p.n_ratings)} movies rated, mean rating ${dec(p.mean_rating)} stars.` : "";
    $("profile-genres").replaceChildren(...(p ? genreList(p.top_genres) : []).map((g) => el("li", {}, g)));
    document.querySelectorAll("#quick-list button").forEach((b) =>
      b.setAttribute("aria-current", String(Number(b.dataset.user) === userId)));
    loadRecommendations();
    loadWatched();
  }

  function renderMeasures() {
    $("measure-group").replaceChildren(...state.info.measures.map((m) => el("button", {
      type: "button", role: "radio", "aria-checked": String(m === state.measure),
      onclick: () => { state.measure = m; renderMeasures(); loadRecommendations(); },
    }, MEASURE_NAMES[m] || m)));
  }
  function updateStepper() {
    $("n-value").textContent = state.n;
    $("n-minus").disabled = state.n <= state.info.n_min;
    $("n-plus").disabled = state.n >= state.info.n_max;
  }
  function showTab(tab) {
    const recs = tab === "recs";
    $("tab-recs").setAttribute("aria-selected", String(recs));
    $("tab-watched").setAttribute("aria-selected", String(!recs));
    $("panel-recs").hidden = !recs;
    $("panel-watched").hidden = recs;
  }

  // Hero preview: five real recommendations for the first profile
  async function loadPreview(userId) {
    try {
      const data = await api(`/api/recommend?user=${userId}&n=5`);
      $("preview-user").textContent = `#${userId}`;
      $("preview-list").replaceChildren(...data.items.map((m) => el("li", {},
        el("b", { title: m.title }, m.title),
        el("span", { class: "score" }, `${dec(m.predicted)} ★`),
        el("small", {}, `Because they liked ${m.because.title}`))));
      $("preview-foot").textContent = `Cosine measure, k = ${state.info.k}, β = ${state.info.beta}. Computed in ${dec(data.elapsed_ms)} ms.`;
    } catch (err) {
      $("preview-foot").textContent = err.message;
    }
  }

  // ================= CHAPTER 3 DATA =================
  function barChart(id, items, { valueOf, labelOf, textOf, highlight = () => true, aria }) {
    const max = Math.max(...items.map(valueOf));
    const chart = $(id);
    chart.setAttribute("aria-label", aria);
    chart.replaceChildren(...items.map((it) => el("div", { class: "chart__col" },
      el("span", { class: "chart__val" }, textOf(it)),
      el("div", { class: `chart__bar${highlight(it) ? "" : " is-dim"}`, style: `height:${(valueOf(it) / max) * 100}%` }))));
    const axis = el("div", { class: "chart__axis", "aria-hidden": "true" }, ...items.map((it) => el("span", {}, labelOf(it))));
    chart.after(axis);
  }

  function renderStats(s) {
    const o = s.overview;
    // stats band
    const band = { n_ratings: fmt.format(o.n_ratings), n_users: fmt.format(o.n_users),
      n_movies_rated: fmt.format(o.n_movies_rated), density: `${dec(o.density, 2)}%` };
    document.querySelectorAll("[data-stat]").forEach((n) => { n.textContent = band[n.dataset.stat]; });

    // Table 3.3
    const files = [
      ["ratings.csv", `userId, movieId, rating, timestamp; ${fmt.format(o.n_ratings)} rows`, "Building the user–movie matrix"],
      ["movies.csv", `movieId, title, genres; ${fmt.format(o.n_movies_catalog)} movies`, "Showing titles and genres"],
      ["links.csv", "movieId, imdbId, tmdbId", "Not used"],
      ["tags.csv", "User-assigned tags", "Not used"],
    ];
    $("table-files").replaceChildren(...files.map((r) => el("tr", {}, ...r.map((c) => el("td", {}, c)))));

    // Train/test split (Section 3.6)
    const sp0 = s.split;
    const splitRows = [
      ["Split", "Random 80/20, random_state = 42"],
      ["Training set", `${fmt.format(sp0.train)} ratings; ${sp0.train_users} users, ${fmt.format(sp0.train_movies)} movies`],
      ["Test set", `${fmt.format(sp0.test)} ratings`],
      ["Test rows for movies absent from training", `${fmt.format(sp0.test_unseen_rows)} (${dec(sp0.test_unseen_share)}%)`],
    ];
    $("table-split").replaceChildren(...splitRows.map(([k, v]) => el("tr", {}, el("td", {}, k), el("td", {}, v))));

    // Table 3.4
    const u = s.per_user, m = s.per_movie;
    const rows = [
      ["Number of ratings", fmt.format(o.n_ratings)],
      ["Number of users", fmt.format(o.n_users)],
      ["Movies with at least one rating", `${fmt.format(o.n_movies_rated)} (${o.n_movies_catalog - o.n_movies_rated} movies have none)`],
      ["Rating scale", `${dec(o.rating_min)} to ${dec(o.rating_max)} stars, in steps of 0.5`],
      ["Density of the user–movie matrix", `${dec(o.density, 2)}% (${dec(100 - o.density, 2)}% of cells empty)`],
      ["Ratings per user", `min ${u.min}; median ${dec(u.median)}; mean ${dec(u.mean)}; max ${fmt.format(u.max)}`],
      ["Ratings per movie", `min ${m.min}; median ${fmt.format(m.median)}; mean ${dec(m.mean)}; max ${m.max}`],
      ["Time span", `${o.time_start} to ${o.time_end}`],
    ];
    $("table-stats").replaceChildren(...rows.map(([k, v]) => el("tr", {}, el("td", {}, k), el("td", {}, v))));

    // Figures 3.1 - 3.3
    barChart("chart-ratings", s.rating_dist, {
      valueOf: (d) => d.count, labelOf: (d) => dec(d.rating), textOf: (d) => `${dec(d.share, 0)}%`,
      highlight: (d) => d.rating >= 3, aria: "Bar chart of the rating distribution from 0.5 to 5 stars",
    });
    $("cap-ratings").replaceChildren(...rich(`Mean **${dec(o.rating_mean, 2)}** stars; **${dec(o.share_ge3)}%** of ratings are 3.0 stars or higher.`));

    const t = s.long_tail;
    barChart("chart-tail", t.buckets, {
      valueOf: (d) => d.count, labelOf: (d) => d.label, textOf: (d) => `${dec(d.share, 0)}%`,
      highlight: (d) => d.label === "1", aria: "Bar chart of movies by number of ratings",
    });
    $("cap-tail").replaceChildren(...rich(`**${fmt.format(t.single)}** movies (${dec(t.single_share)}%) have a single rating; the 10% most popular movies receive **${dec(t.top10_share)}%** of all ratings. The horizontal axis is ratings per movie.`));

    const sp = s.support;
    barChart("chart-support", sp.buckets, {
      valueOf: (d) => d.share, labelOf: (d) => d.label, textOf: (d) => `${dec(d.share)}%`,
      highlight: (d) => d.label === "1", aria: "Bar chart of movie pairs by number of shared raters",
    });
    $("cap-support").replaceChildren(...rich(`Of the **${dec(sp.pairs / 1e6)} million** movie pairs with a shared rater (training set), **${dec(sp.buckets[0].share)}%** share exactly one.`));

    // Table 3.5
    const decisions = [
      [`${dec(o.density, 2)}%`, "of matrix cells hold a value", "Store float32 arrays and compute everything with **matrix multiplication**, with no loop over user–movie pairs."],
      [`${dec(sp.buckets[0].share)}%`, "of movie pairs share only one rater", "**Shrink** similarities by the number of shared raters (β) and keep only the **k strongest neighbours**."],
      [`${dec(t.single_share)}%`, "of movies have a single rating", "Rank the Top-N list by **summed similarity** instead of predicted rating; drop movies with fewer than m ratings."],
      [`${u.min} ratings`, "is the minimum per user", "A random **80/20** split by rating still keeps every user in the training set."],
      [fmt.format(s.split.test_unseen_rows), `test rows (${dec(s.split.test_unseen_share)}%) belong to movies absent from the training set`, "Use the **user mean** as a fallback and report **Coverage** separately."],
      ["Each user", "has a personal rating level, strict or lenient", "For Pearson, use the **mean-centred** prediction formula."],
      [`${dec(o.rating_mean, 2)} stars`, "is the mean rating, skewed high", "Treat movies rated **4.0 stars** or higher as relevant for Precision@K; always compare with **baselines**."],
    ];
    $("decisions").replaceChildren(...decisions.map(([num, fact, todo]) => el("li", { class: "decision" },
      el("p", { class: "decision__fact" }, el("strong", {}, num), fact),
      el("p", { class: "decision__do" }, ...rich(todo)))));
  }

  function renderParams(info) {
    const rows = [
      ["MEASURE", info.measure, "Default similarity measure of the app"],
      ["K_NEIGHBORS", info.k ?? "all", "Neighbours kept per movie (k)"],
      ["BETA", info.beta, "Shrinkage threshold on shared raters (β)"],
      ["MIN_ITEM_RATINGS", info.min_item_ratings, "Minimum ratings for a movie to be recommended (m)"],
      ["EVAL_K", info.eval_k, "List length for Precision@K and Recall@K"],
      ["RELEVANT_THRESHOLD", dec(info.relevant_threshold), "Rating threshold for a movie to count as relevant"],
      ["TEST_SIZE, RANDOM_STATE", `${dec(info.test_size)}; ${info.random_state}`, "Test set fraction and random seed"],
    ];
    $("table-params").replaceChildren(...rows.map(([k, v, d]) => el("tr", {}, el("td", {}, el("code", {}, k)), el("td", {}, String(v)), el("td", {}, d))));
    $("config-note").textContent = `Item-based CF, k = ${info.k}, β = ${info.beta}. Only movies with at least ${info.min_item_ratings} ratings are recommended.`;
  }

  // ================= CHAPTER 5 RESULTS =================
  const RESULT_NAMES = {
    B1: ["B1 Global mean", ""], B2: ["B2 User mean", ""], B3: ["B3 Most popular", ""],
    E1: ["E1 Cosine, first version", "all neighbours, no shrinkage, ranked by predicted rating"],
    E2: ["E2 Cosine", "main configuration, k = 50, β = 10"], E3: ["E3 Pearson", "k = 50, β = 10"],
    E4: ["E4 Jaccard", "k = 50, β = 10"], E5: ["E5 User-based Cosine", "for comparison"],
  };
  async function loadResults() {
    try {
      const data = await api("/api/results");
      if (!data.available) {
        $("results-note").textContent = "No results yet. Run python evaluate.py, then reload the page.";
        return;
      }
      const num = (v) => (v === null || v === undefined ? "–" : dec(v, 4));
      $("table-results").replaceChildren(...data.rows.map((r) => {
        const [name, note] = RESULT_NAMES[r.id] || [r.id, ""];
        return el("tr", { class: r.id === "E2" ? "is-main" : "" },
          el("td", {}, name, note ? el("small", {}, note) : null),
          ...[r.rmse, r.mae, r.coverage, r.precision, r.recall].map((v) => el("td", {}, num(v))));
      }));
      const by = Object.fromEntries(data.rows.map((r) => [r.id, r]));
      if (by.E2 && by.B2 && by.B3) {
        const gain = (by.E2.precision / by.B3.precision - 1) * 100;
        const cut = (1 - by.E2.rmse / by.B2.rmse) * 100;
        $("highlights").replaceChildren(
          el("div", { class: "highlight" }, el("strong", {}, dec(by.E2.precision, 4)), el("span", {}, `Precision@5 of Cosine, ${dec(gain, 0)}% higher than recommending the most popular movies`)),
          el("div", { class: "highlight" }, el("strong", {}, dec(by.E2.rmse, 4)), el("span", {}, `RMSE of Cosine, ${dec(cut)}% lower than predicting with the user mean`)),
          el("div", { class: "highlight" }, el("strong", {}, "16/16"), el("span", {}, "automated tests pass; recommending for one user takes only a few milliseconds")));
      }
    } catch (err) {
      $("results-note").textContent = err.message;
    }
  }

  // Highlight the section in view on the navigation bar
  function trackSections() {
    const links = [...document.querySelectorAll(".nav__links a")];
    const map = new Map(links.map((a) => [a.getAttribute("href").slice(1), a]));
    const obs = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) links.forEach((a) => a.classList.toggle("is-active", a === map.get(e.target.id)));
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    map.forEach((_, id) => { const s = $(id); if (s) obs.observe(s); });
  }

  // ================= START-UP =================
  async function init() {
    trackSections();
    try {
      state.info = await api("/api/info");
      state.n = state.info.n_default;
      state.measure = state.info.measure;
      renderParams(state.info);
      state.profiles = (await api(`/api/profiles?limit=${state.info.n_users}`)).profiles;
      state.profiles.forEach((p) => state.byId.set(p.userId, p));
      renderOptions();
      renderMeasures();
      updateStepper();
      renderQuick();
      chooseUser(state.profiles[0].userId);
      loadPreview(state.profiles[0].userId);
      renderStats(await api("/api/stats"));
      loadResults();
    } catch (err) {
      setStatus("rec-status", `${err.message} Run python app.py, then reload the page.`, true);
    }

    $("profile-input").addEventListener("input", (e) => { e.target.value = e.target.value.replace(/\D/g, ""); });
    $("profile-input").addEventListener("keydown", (e) => { if (e.key === "Enter") submitProfile(); });
    $("profile-input").addEventListener("change", () => { if (state.byId.has(Number($("profile-input").value))) submitProfile(); });
    $("profile-go").addEventListener("click", submitProfile);
    $("n-minus").addEventListener("click", () => { state.n -= 1; updateStepper(); loadRecommendations(); });
    $("n-plus").addEventListener("click", () => { state.n += 1; updateStepper(); loadRecommendations(); });
    $("tab-recs").addEventListener("click", () => showTab("recs"));
    $("tab-watched").addEventListener("click", () => showTab("watched"));
  }

  init();
})();
