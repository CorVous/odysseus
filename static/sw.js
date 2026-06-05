// static/sw.js — Odysseus PWA Service Worker
// Strategy:
//   - HTML (navigation): stale-while-revalidate. Instant open from cache,
//     background refresh so the next open has latest HTML.
//   - JS/CSS (/static/*.js|.css): network-first, cache fallback for offline.
//     (So code/style edits show up on a normal reload, no manual cache clear.)
//   - Other static assets (images/fonts/libs): cache-first with bg refresh.
//   - API / non-GET: never cached.
// Bump CACHE_NAME whenever the precache list or SW logic changes.
// v327: force re-precache so clients pick up the updated chat.js + sessions.js
// (resume-stream-on-reload fix) — the unversioned sessions.js was being served
// stale from an older SW cache.
// v328: drop the chat.js self-trigger (it raced with the history re-render and
// wiped the resume bubble); resume is driven solely by sessions._checkServerStream.
// v337: bump to force re-precache so clients pick up the latest static assets.
// v338: add "Swipe to Open Sidebar" toggle (index.html + sidebar-layout.js).
// v339: resume rerender preserves scroll (no more snap to last user message) and
// thinking expand/collapse state (no more wrong-box / every-tick force-expand).
// v340: actually land the "Swipe to Open Sidebar" toggle UI + guard (the v338
// comment referenced it but the index.html/sidebar-layout.js changes weren't
// merged until now).
// v341: kind-prefixed jump-link routing fix (document-link-bounce-to-home).
// v342: open orphaned library docs directly by id (library-open-orphaned-documents).
// v344: agent AI tab — Max rounds input + Smart stop toggle.
// v345: thinking-box fixes for multi-round/resumed turns —
//       (1) unique section ids (markdown.js + chat.js live render) so the toggle
//           handler resolves the box you clicked, not the first one;
//       (2) kill the expand transition on resumed boxes so the open animation
//           doesn't re-fire on every rerender tick;
//       (3) the hash-restore observer skips resume-rendered turns (resume restores
//           its own state by index) — stops per-tick re-expansion + same-content
//           cross-firing (clicking one box opening another);
//       (4) resume updates the body in place on plain text deltas (current round
//           for tool turns, the merged bubble for no-tools turns) instead of
//           rebuilding the whole turn every 140ms — fixes bursty, batched UI
//           updates; full rebuild only at structural (round/tool) boundaries.
// v346: resume scroll fixes — first paint after reattach snaps to the live turn
//       (was leaving you parked at your last message); streaming thinking box is
//       pinned to its latest line (innerHTML replacement was yanking it to the top
//       and fighting scroll-down).
// v347: resume paints the buffered backlog in ONE render (backend emits a
//       `resume_synced` marker after replay; client accumulates until then)
//       instead of a rapid burst of incremental updates, then streams the live tail.
// v348: resume can't get stuck on "Reconnecting…" if the marker never arrives
//       (e.g. backend not yet restarted) — a grace-period fallback forces the
//       one-shot paint and switches to live rendering.
// v349: that fallback is now a debounced idle timer (reset per event) so it fires
//       only after the backlog burst drains, never mid-burst — fixes per-event
//       re-renders/freeze when the marker is absent on a long playback.
const CACHE_NAME = 'odysseus-v349';

// Core shell precached on install so repeat opens are instant without any
// network wait. Keep this list in sync with the <script type="module"> tags
// and <link rel="stylesheet"> in index.html.
const PRECACHE = [
  '/',
  '/static/style.css',
  '/static/app.js',
  '/static/js/storage.js',
  '/static/js/ui.js',
  '/static/js/markdown.js',
  '/static/js/dragSort.js',
  '/static/js/sessions.js',
  '/static/js/memory.js',
  '/static/js/skills.js',
  '/static/js/tourHints.js',
  '/static/js/fileHandler.js',
  '/static/js/voiceRecorder.js',
  '/static/js/models.js',
  '/static/js/rag.js',
  '/static/js/presets.js',
  '/static/js/search.js',
  '/static/js/spinner.js',
  '/static/js/tts-ai.js',
  '/static/js/document.js',
  '/static/js/gallery.js',
  '/static/js/chatRenderer.js',
  '/static/js/codeRunner.js',
  '/static/js/chatStream.js',
  '/static/js/chat.js',
  '/static/js/cookbook.js',
  '/static/js/search-chat.js',
  '/static/js/compare/index.js',
  '/static/js/theme.js',
  '/static/js/censor.js',
  '/static/js/settings.js',
  '/static/js/admin.js',
  '/static/js/init.js',
  '/static/js/slashCommands.js',
  '/static/js/emailInbox.js',
  '/static/js/emailLibrary/utils.js',
  '/static/js/emailLibrary/signatureFold.js',
  '/static/js/emailLibrary/state.js',
  '/static/js/notes.js',
  '/static/js/tasks.js',
  '/static/js/calendar.js',
  '/static/js/calendar/utils.js',
  '/static/js/calendar/reminders.js',
  '/static/js/group.js',
  '/static/js/keyboard-shortcuts.js',
  '/static/js/sidebar-layout.js',
  '/static/js/section-management.js',
  '/static/lib/highlight.min.js',
];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE_NAME).then(cache =>
      // addAll is atomic — if any item fails, none are cached. Use individual
      // puts so a single 404 can't block the whole install.
      Promise.all(
        PRECACHE.map(url =>
          fetch(url, { cache: 'reload' })
            .then(res => res.ok ? cache.put(url, res) : null)
            .catch(() => null)
        )
      )
    )
  );
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);

  // Never touch API calls or non-GET.
  if (url.pathname.startsWith('/api/') || e.request.method !== 'GET') return;

  // HTML navigation: stale-while-revalidate the app shell — but ONLY for the
  // SPA root. Other navigations (e.g. a deep-linked /static/*.html page) must
  // go to the network/static handlers below; otherwise every navigation was
  // served the app index, replacing the page the user actually asked for.
  if (e.request.mode === 'navigate' && url.pathname === '/') {
    e.respondWith(
      caches.open(CACHE_NAME).then(async cache => {
        const cached = await cache.match('/');
        const network = fetch(e.request).then(res => {
          if (res && res.ok) cache.put('/', res.clone());
          return res;
        }).catch(() => cached);
        return cached || network;
      })
    );
    return;
  }

  // JS/CSS: network-first — always try the network so code/style edits show up
  // on a normal reload; fall back to cache only when offline.
  if (url.pathname.startsWith('/static/') && /\.(js|css)(\?|$)/.test(url.pathname + url.search)) {
    e.respondWith(
      fetch(e.request).then(res => {
        if (res && res.ok) {
          const copy = res.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(e.request, copy));
        }
        return res;
      }).catch(() => caches.match(e.request))
    );
    return;
  }

  // Other static assets (images, fonts, libs): cache-first with background refresh.
  if (url.pathname.startsWith('/static/')) {
    e.respondWith(
      caches.open(CACHE_NAME).then(async cache => {
        const cached = await cache.match(e.request);
        const fetching = fetch(e.request).then(res => {
          if (res && res.ok) cache.put(e.request, res.clone());
          return res;
        }).catch(() => cached);
        return cached || fetching;
      })
    );
    return;
  }
});
