// 资源获取子页渲染。完全由 participant_kit/manifest.json 驱动 —— 新增/调整资源只改 manifest，
// 不动本文件。拉取套路同 board.js（静态 fetch + ?t= 破缓存），无后端。
// 资源类型：doc(markdown 内联渲染) / code(代码块) / usd(整包 zip 下载)。
(function () {
  'use strict';

  var cfg = window.BOARD_CONFIG || {};
  var BASE = cfg.KIT_BASE || './participant_kit/';
  var MANIFEST = BASE + 'manifest.json';
  var TYPE_LABEL = { doc: '文档', code: '代码', usd: 'USD' };

  function esc(s) {
    var d = document.createElement('div');
    d.textContent = String(s == null ? '' : s);
    return d.innerHTML;
  }
  function bust(url) { return url + (url.indexOf('?') < 0 ? '?' : '&') + 't=' + Date.now(); }
  function zipOf(bundle) { return String(bundle).replace(/\/+$/, '') + '.zip'; }  // 文件夹 → 同名 zip

  function fetchJson(url) {
    return fetch(bust(url)).then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; });
  }
  function fetchText(url) {
    return fetch(bust(url)).then(function (r) { return r.ok ? r.text() : null; }).catch(function () { return null; });
  }

  function renderMeta(m) {
    var t = document.getElementById('kit-title');
    if (t && (m.title_zh || m.title)) t.textContent = m.title_zh || m.title;
    var d = document.getElementById('kit-desc');
    if (d && (m.description_zh || m.description)) d.textContent = m.description_zh || m.description;
    var meta = document.getElementById('kit-meta');
    if (meta) {
      var bits = [];
      if (m.version) bits.push('v' + m.version);
      if (m.updated_at) bits.push('更新 ' + m.updated_at);
      meta.textContent = bits.join(' · ');
    }
  }

  // 链接分组：manifest 每条链接可带 group（tron / humanoid），按下列顺序分区加小标题；
  // 未识别 group 落「其他」；全部无 group 时退回平铺（向后兼容旧 manifest）。
  var LINK_GROUPS = [
    { key: 'tron', label: 'TRON 2 赛题' },
    { key: 'humanoid', label: 'Oli 人形赛题' }
  ];

  function linkCard(l) {
    return '<a class="card kit-link" href="' + esc(l.url) + '" target="_blank" rel="noopener">' +
      '<h3>' + esc(l.name_zh || l.name) + ' ↗</h3>' +
      '<p>' + esc(l.desc_zh || l.desc || '') + '</p></a>';
  }

  function renderLinks(links) {
    var box = document.getElementById('kit-links');
    if (!box) return;
    if (!links || !links.length) { box.innerHTML = '<p class="kit-fallback">（暂无外部链接）</p>'; return; }
    if (!links.some(function (l) { return l.group; })) {   // 无分组信息 → 平铺
      box.innerHTML = links.map(linkCard).join('');
      return;
    }
    var known = {}, html = '';
    LINK_GROUPS.forEach(function (g) {
      known[g.key] = true;
      var items = links.filter(function (l) { return l.group === g.key; });
      if (items.length) html += '<h4 class="link-group-title">' + esc(g.label) + '</h4>' + items.map(linkCard).join('');
    });
    var others = links.filter(function (l) { return !l.group || !known[l.group]; });
    if (others.length) html += '<h4 class="link-group-title">其他</h4>' + others.map(linkCard).join('');
    box.innerHTML = html;
  }

  function renderResources(resources) {
    var box = document.getElementById('kit-resources');
    if (!box) return;
    box.innerHTML = '';
    if (!resources || !resources.length) { box.innerHTML = '<p class="kit-fallback">（暂无资源）</p>'; return; }
    resources.forEach(function (r) { box.appendChild(resourceCard(r)); });
  }

  function isFolderPath(p) { return !!p && p.slice(-1) === '/'; }

  function resourceCard(r) {
    var path = r.path || '';
    var folder = r.bundle || (isFolderPath(path) ? path : '');     // 文件夹资源 → zip 下载
    var isZip = !!folder;
    var dl = BASE + (isZip ? zipOf(folder) : path);
    // 预览的具体文件：优先 manifest 的 entry（文件夹用），否则用单文件 path
    var preview = r.entry || (path && !isFolderPath(path) ? path : '');
    var previewable = (r.type === 'doc' || r.type === 'code') && !!preview;

    var card = document.createElement('div');
    card.className = 'card kit-card';
    var size = r.size_mb ? '<span class="kit-size">' + esc(r.size_mb) + ' MB</span>' : '';
    var badge = '<span class="kit-badge">' + esc(TYPE_LABEL[r.type] || r.type || '文件') + '</span>';
    var actions = '';
    if (previewable) actions += '<button class="btn kit-view" type="button">查看</button>';
    actions += '<a class="btn' + (isZip ? ' primary' : '') + '" download href="' + esc(dl) + '">' +
      (isZip ? '下载 zip' : '下载') + '</a>';

    card.innerHTML =
      '<div class="kit-card-head"><h3>' + esc(r.name_zh || r.name) + ' ' + badge + ' ' + size + '</h3></div>' +
      '<p>' + esc(r.desc_zh || r.desc || '') + '</p>' +
      '<div class="kit-actions">' + actions + '</div>' +
      '<div class="kit-doc" hidden></div>';

    if (previewable) wireView(card, preview);
    return card;
  }

  // 「查看」懒加载 + 折叠：首次点开才 fetch 原文；.md/.markdown 走 marked 渲染，其余进 <pre><code>。
  function wireView(card, previewPath) {
    var btn = card.querySelector('.kit-view');
    var docBox = card.querySelector('.kit-doc');
    var loaded = false;
    var isMd = /\.(md|markdown)$/i.test(previewPath);
    btn.addEventListener('click', function () {
      if (!docBox.hasAttribute('hidden')) {
        docBox.setAttribute('hidden', '');
        btn.textContent = '查看';
        return;
      }
      docBox.removeAttribute('hidden');
      btn.textContent = '收起';
      if (loaded) return;
      loaded = true;
      docBox.innerHTML = '<p class="kit-fallback">加载中…</p>';
      fetchText(BASE + previewPath).then(function (txt) {
        if (txt == null) {
          docBox.innerHTML = '<p class="kit-fallback">资源暂不可用，请用「下载」获取</p>';
          loaded = false;  // 允许重试
          return;
        }
        if (isMd && window.marked && typeof window.marked.parse === 'function') {
          docBox.innerHTML = window.marked.parse(txt);
        } else {
          docBox.innerHTML = '<pre><code>' + esc(txt) + '</code></pre>';
        }
      });
    });
  }

  function load() {
    fetchJson(MANIFEST).then(function (m) {
      if (!m) {
        var l = document.getElementById('kit-links');
        if (l) l.innerHTML = '<p class="kit-fallback">资源清单加载失败，请稍后刷新重试</p>';
        var rb = document.getElementById('kit-resources');
        if (rb) rb.innerHTML = '';
        return;
      }
      renderMeta(m);
      renderLinks(m.links);
      renderResources(m.resources);
    });
  }

  window.addEventListener('DOMContentLoaded', load);
})();
