// 合并榜单渲染（TronCamp Task C + Humanoid Task F，同站同榜、统一排名）。
// 两源本地读取：tron ./data/leaderboard.json，人形 ./data/humanoid.json。各赛题 breakdown 不混。
// 列：# / Token / 赛题 / 机型 / 总分(进度条 + 完赛·用时)。身份只显示 token 尾号、不显示队名。
// 机型权重(wfyg 轮式 ×0.8)已由 Worker 算进每行 total；前端只按 total 排名，权重仅用于展示(×0.8 标记)。
(function () {
  'use strict';

  var cfg = window.BOARD_CONFIG || {};
  var TRON_URL = cfg.TRON_DATA_URL || './data/leaderboard.json';
  var HUM_URL = cfg.HUMANOID_DATA_URL || './data/humanoid.json';
  var REFRESH = (cfg.REFRESH_SECONDS || 60) * 1000;
  var BOARD = 'dev';  // 实际值在 DOMContentLoaded 时从 <body data-board> 读取
  // 终榜冻结时刻：此后 final 页固定展示 dev 榜数据（提交 24:00 截止，评测队列结算到 04:00）。
  var FREEZE_MS = new Date(cfg.FINAL_FREEZE_AT || '2026-07-21T04:00:00+08:00').getTime();

  function esc(s) {
    var d = document.createElement('div');
    d.textContent = String(s);
    return d.innerHTML;
  }

  function fmt(n, d) {
    if (n === null || n === undefined) return '—';
    return Number(n).toFixed(d === undefined ? 1 : d);
  }

  // 身份：优先显示选手令牌【明文后 6 位】(display)；无则回退 token_id 尾 6；再无回退队名。
  function tokenSuffix(r) {
    if (r.display) return '··' + esc(String(r.display).slice(-6));
    if (r.token) return '··' + esc(String(r.token).slice(-6));
    if (r.team) return esc(r.team);
    return '<span class="dimcell">—</span>';
  }

  // 赛题标签：tron -> Tron；humanoid -> 人形。
  function competitionTag(c) {
    return c === 'tron'
      ? '<span class="vchip">Tron</span>'
      : '<span class="vchip">人形</span>';
  }

  // 机型标签：直接显示型号 + 中文构型（sfyg_tron2a（腿式）/ wfyg_tron2a（轮式）/ oli（人形））。
  // robot_type 优先，缺省按 variant 推断；轮式(得分权重 0.8)附 ×0.8 让分标，-Orig 传感变体标注。
  var MODEL_LABEL = {
    sfyg_tron2a: 'sfyg_tron2a（腿式）',
    wfyg_tron2a: 'wfyg_tron2a（轮式）',
    oli: 'oli（人形）'
  };
  function modelTag(robot, variant, weight) {
    var label = MODEL_LABEL[robot];
    if (!label && variant) label = /Wheel/.test(variant) ? MODEL_LABEL.wfyg_tron2a : MODEL_LABEL.sfyg_tron2a;
    if (!label) return '<span class="dimcell">—</span>';
    var isWheel = robot === 'wfyg_tron2a' || /Wheel/.test(variant || '');
    var orig = /Orig$/.test(variant || '') ? '<span class="vtag">ORIG</span>' : '';
    var w = (weight != null) ? weight : (isWheel ? 0.8 : 1.0);
    var wtag = (w !== 1) ? '<span class="vtag">×' + w + '</span>' : '';
    return '<span class="vchip">' + esc(label) + '</span>' + orig + wtag;
  }

  function totalCell(r) {
    if (r.total === null || r.total === undefined) {
      return '<td class="c-t3"><span class="dimcell">未上场</span></td>';
    }
    var w = Math.max(2, Math.min(100, r.total));
    var fin = r.finished ? '完赛' : '未完赛';
    var t = (r.elapsed != null) ? ' · ' + fmt(r.elapsed, 0) + 's' : '';
    var sub = '<span class="t3sub">' + fin + t + '</span>';
    return '<td class="c-t3"><div class="t3wrap">' +
      '<span class="t3num">' + fmt(r.total) + '</span>' + sub +
      '<span class="t3bar"><i style="width:' + w + '%"></i></span></div></td>';
  }

  function rowHtml(r) {
    var cls = r.rank <= 3 ? ' top top' + r.rank : '';
    return '<tr class="brow' + cls + '">' +
      '<td class="c-rank">' + r.rank + '</td>' +
      '<td class="c-team">' + tokenSuffix(r) + '</td>' +
      '<td class="c-gate">' + competitionTag(r.competition) + '</td>' +
      '<td class="c-gate c-model">' + modelTag(r.robot_type, r.variant, r.weight) + '</td>' +
      totalCell(r) +
      '</tr>';
  }

  // 归一化：抽排名/展示要用的字段（total 已含机型权重）。各赛题 breakdown 原样保留、互不糅合。
  function normalize(raw, competition) {
    var b = raw.breakdown || {};
    return {
      token: raw.token,
      display: raw.display,
      team: raw.team,
      competition: competition,
      robot_type: raw.robot_type,
      variant: competition === 'tron' ? raw.variant : null,
      weight: raw.weight,
      total: raw.total,
      finished: !!b.finished,
      elapsed: (b.elapsed_finish != null) ? b.elapsed_finish : null,
      best_submit: raw.best_submit || raw.last_submit || ''
    };
  }

  // 按当前 BOARD(dev/final) 从一份数据里选行。
  // dev: data.dev。final: 解锁且有终榜用 final；冻结时刻后用 dev（冻结展示）；
  // 否则该源锁定(返回 null)——含截止后至冻结前的评测结算窗口。
  function pickRows(data, frozen) {
    if (!data) return [];
    if (BOARD === 'final') {
      if (data.final_unlocked && data.final && data.final.length) return data.final;
      if (frozen) return data.dev || [];
      return null;  // locked
    }
    return data.dev || [];
  }

  var countdownTimer = null;

  function renderCountdown(deadline) {
    var el = document.getElementById('countdown');
    if (!el) return;
    if (!deadline) { el.textContent = '—'; return; }
    var end = new Date(deadline).getTime();
    function tick() {
      var ms = end - Date.now();
      if (ms <= 0) {
        el.textContent = Date.now() >= FREEZE_MS
          ? '已截止 · FINAL 榜为最终成绩'
          : '已截止 · 评测结算中';
        el.classList.add('over');
        return;
      }
      var s = Math.floor(ms / 1000);
      var d = Math.floor(s / 86400);
      var pad = function (n) { return String(n).padStart(2, '0'); };
      el.textContent = '距截止 ' + d + ' 天 ' + pad(Math.floor(s % 86400 / 3600)) +
        ':' + pad(Math.floor(s % 3600 / 60)) + ':' + pad(s % 60);
    }
    if (countdownTimer) clearInterval(countdownTimer);
    tick();
    countdownTimer = setInterval(tick, 1000);
  }

  function render(tronData, humData) {
    var updated = document.getElementById('updated');
    if (updated) {
      var g = [tronData && tronData.generated_at, humData && humData.generated_at].filter(Boolean);
      updated.textContent = '更新于 ' + (g.length ? g.sort()[g.length - 1] : '—');
    }

    // 截止时间取任一源（两赛题截止一致）。
    var deadline = (humData && humData.deadline) || (tronData && tronData.deadline) || null;
    renderCountdown(deadline);
    var frozen = Date.now() >= FREEZE_MS;

    var locked = document.getElementById('locked');
    var table = document.getElementById('board-table');
    var empty = document.getElementById('empty');

    var tronRows = pickRows(tronData, frozen);
    var humRows = pickRows(humData, frozen);

    // final 板：两源都锁定 → 显示赛末公布（含截止后至冻结前的结算窗口）。
    if (BOARD === 'final' && tronRows === null && humRows === null) {
      if (locked) locked.hidden = false;
      if (table) table.hidden = true;
      if (empty) empty.hidden = true;
      return;
    }
    if (locked) locked.hidden = true;
    tronRows = tronRows || [];
    humRows = humRows || [];

    // 合并 + 统一排名（与 worker/writeback.py:_rank_key 同一规则）：total 降序（已含权重）
    // → 完赛优先 → 都完赛比用时（未完赛 elapsed 是各赛道封顶值 600/720，不可比）
    // → best_submit 先达到者在前（老数据回退 last_submit）。前端重排 rank。
    var merged = tronRows.map(function (r) { return normalize(r, 'tron'); })
      .concat(humRows.map(function (r) { return normalize(r, 'humanoid'); }))
      .filter(function (r) { return r.total !== null && r.total !== undefined; });
    merged.sort(function (a, b) {
      var d = (b.total || 0) - (a.total || 0);
      if (d) return d;
      d = (b.finished ? 1 : 0) - (a.finished ? 1 : 0);
      if (d) return d;
      if (a.finished && b.finished) {
        d = (a.elapsed || 0) - (b.elapsed || 0);
        if (d) return d;
      }
      return a.best_submit < b.best_submit ? -1 : a.best_submit > b.best_submit ? 1 : 0;
    });
    merged.forEach(function (r, i) { r.rank = i + 1; });

    if (!merged.length) {
      if (table) table.hidden = true;
      if (empty) empty.hidden = false;
      return;
    }
    table.hidden = false;
    if (empty) empty.hidden = true;
    table.querySelector('tbody').innerHTML = merged.map(rowHtml).join('');
  }

  function fetchSrc(url) {
    return fetch(url + '?t=' + Date.now())
      .then(function (r) { return r.ok ? r.json() : null; })
      .catch(function () { return null; });  // 一源挂掉返回 null，另一源照常渲染
  }

  function load() {
    Promise.all([fetchSrc(TRON_URL), fetchSrc(HUM_URL)])
      .then(function (res) { render(res[0], res[1]); });
  }

  window.addEventListener('DOMContentLoaded', function () {
    BOARD = document.body.dataset.board || 'dev';
    load();
    setInterval(load, REFRESH);
  });
})();
