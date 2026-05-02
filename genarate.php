<?php

$readme = file_get_contents(__DIR__ . '/README.md');

function parseGitHubRepo(string $url): ?array
{
    $parts = parse_url($url);

    if (!isset($parts['host']) || strtolower($parts['host']) !== 'github.com' || !isset($parts['path'])) {
        return null;
    }

    $path = trim($parts['path'], '/');
    $segments = explode('/', $path);

    if (count($segments) < 2) {
        return null;
    }

    $owner = $segments[0];
    $repo = preg_replace('/\.git$/', '', $segments[1]);

    if ($owner === '' || $repo === '') {
        return null;
    }

    return [$owner, $repo];
}

function fetchGitHubStars(string $owner, string $repo): ?int
{
    $apiUrl = "https://api.github.com/repos/{$owner}/{$repo}";
    $context = stream_context_create([
        'http' => [
            'method' => 'GET',
            'header' => [
                'User-Agent: awesome-php-ml-docs-generator',
                'Accept: application/vnd.github+json',
            ],
            'timeout' => 10,
        ],
    ]);

    $response = @file_get_contents($apiUrl, false, $context);

    if ($response === false) {
        return null;
    }

    $json = json_decode($response, true);

    if (!is_array($json) || !isset($json['stargazers_count'])) {
        return null;
    }

    return (int) $json['stargazers_count'];
}

preg_match_all(
    '/^- (🌟|🧪|⚠️)?\s*\[([^\]]+)\]\((https?:\/\/[^)\s"]+)(?:\s+"[^"]*")?\)\s*[-–]\s*(.+)$/mu',
    $readme,
    $matches,
    PREG_SET_ORDER
);

$items = [];
$starsCache = [];
$currentCategory = 'General';
$resourceCount = 0;

$lines = explode("\n", $readme);

foreach ($lines as $line) {
    if (preg_match('/^##\s+(.+)/', $line, $m) && !in_array(trim($m[1]), ['Contents', 'Requirements', 'Legend', 'Resources', 'Contributing', 'License'])) {
        $currentCategory = trim($m[1]);
    }

    if (preg_match('/^- (🌟|🧪|⚠️)?\s*\[([^\]]+)\]\((https?:\/\/[^)\s"]+)(?:\s+"[^"]*")?\)\s*[-–]\s*(.+)$/u', $line, $m)) {
        $resourceCount++;
        $name = $m[2];
        $url = $m[3];
        $stars = null;
        $repoData = parseGitHubRepo($url);

        echo "[{$resourceCount}] Processing: {$name}";

        if ($repoData !== null) {
            [$owner, $repo] = $repoData;
            $repoKey = strtolower($owner . '/' . $repo);

            if (!array_key_exists($repoKey, $starsCache)) {
                echo " | fetching stars for {$owner}/{$repo}";
                $starsCache[$repoKey] = fetchGitHubStars($owner, $repo);
            } else {
                echo " | using cached stars for {$owner}/{$repo}";
            }

            $stars = $starsCache[$repoKey];

            if ($stars !== null) {
                echo " | ⭐ {$stars}";
            } else {
                echo " | stars unavailable";
            }
        } else {
            echo " | non-GitHub link, stars skipped";
        }

        echo "\n";

        $description = preg_replace('/!\[[^\]]*\]\([^)]*\)\s*/u', '', $m[4]);
        $description = trim($description);

        $items[] = [
            'badge' => $m[1] ?: '',
            'name' => $name,
            'url' => $url,
            'description' => $description,
            'category' => $currentCategory,
            'stars' => $stars,
        ];
    }
}

$categories = array_values(array_unique(array_column($items, 'category')));

$data = json_encode($items, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);

$html = <<<HTML
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Awesome PHP Machine Learning & AI</title>
<style>
:root {
  --bg: #0f172a;
  --card: #111827;
  --muted: #94a3b8;
  --text: #e5e7eb;
  --accent: #38bdf8;
  --border: #1f2937;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: radial-gradient(circle at top, #1e3a8a 0, var(--bg) 32%);
  color: var(--text);
  scrollbar-width: thin;
  scrollbar-color: #475569 #020617;
}
::-webkit-scrollbar {
  width: 6px;
}
::-webkit-scrollbar-track {
  background: #020617;
}
::-webkit-scrollbar-thumb {
  background: #475569;
  border-radius: 999px;
}
::-webkit-scrollbar-thumb:hover {
  background: #64748b;
}
.top-link {
  position: fixed;
  top: 18px;
  right: 18px;
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.875rem;
  color: #9ca3af;
  border: 1px solid #374151;
  border-radius: 10px;
  padding: 8px 12px;
  transition: color .2s ease, border-color .2s ease;
  text-decoration: none;
  background: rgba(2, 6, 23, .65);
  backdrop-filter: blur(4px);
}
.top-link:hover {
  color: #ffffff;
  border-color: #6b7280;
}
.top-link svg {
  width: 16px;
  height: 16px;
  fill: currentColor;
}
header {
  padding: 72px 20px 36px;
  text-align: center;
}
h1 {
  font-size: clamp(2.4rem, 6vw, 4.5rem);
  margin: 0;
}
.subtitle {
  color: var(--muted);
  font-size: 1.2rem;
  margin-top: 12px;
}
.container {
  max-width: 1180px;
  margin: 0 auto;
  padding: 20px;
}
.stats {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 14px;
  margin-bottom: 24px;
}
.stat {
  background: rgba(17, 24, 39, .8);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 18px;
  text-align: center;
}
.stat strong {
  display: block;
  font-size: 1.8rem;
}
.controls {
  display: flex;
  gap: 12px;
  margin-bottom: 24px;
  flex-wrap: wrap;
}
input, select {
  background: #020617;
  border: 1px solid var(--border);
  color: var(--text);
  border-radius: 14px;
  padding: 14px 16px;
  font-size: 1rem;
}
input { flex: 1; min-width: 260px; }
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 18px;
}
.card {
  background: linear-gradient(180deg, rgba(17,24,39,.95), rgba(15,23,42,.95));
  border: 1px solid var(--border);
  border-radius: 22px;
  padding: 20px;
  box-shadow: 0 18px 40px rgba(0,0,0,.22);
}
.card:hover {
  border-color: var(--accent);
  transform: translateY(-2px);
  transition: .18s ease;
}
.card h2 {
  font-size: 1.15rem;
  margin: 0 0 10px;
}
.card a {
  color: var(--text);
  text-decoration: none;
}
.card a:hover { color: var(--accent); }
.desc {
  color: var(--muted);
  line-height: 1.5;
  min-height: 72px;
}
.meta {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  margin-top: 18px;
  align-items: center;
}
.tag {
  font-size: .8rem;
  padding: 6px 10px;
  background: #020617;
  border: 1px solid var(--border);
  border-radius: 999px;
  color: var(--muted);
}
.badge {
  font-size: 1.15rem;
}
footer {
  color: var(--muted);
  text-align: center;
  padding: 40px 20px;
}
.empty {
  display: none;
  text-align: center;
  color: var(--muted);
  padding: 40px;
}
@media (max-width: 720px) {
  .stats { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<a href="https://github.com/apphp/awesome-php-ml" target="_blank" rel="noopener" class="top-link">
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"></path>
  </svg>
  GitHub
</a>
<header>
  <h1>Awesome PHP ML & AI</h1>
  <p class="subtitle">Curated PHP libraries for Machine Learning, AI, NLP, LLMs, agents, RAG and data science.</p>
</header>

<main class="container">
  <section class="stats">
    <div class="stat"><strong id="total">0</strong>Libraries</div>
    <div class="stat"><strong id="categories">0</strong>Categories</div>
    <div class="stat"><strong>PHP</strong>AI Ecosystem</div>
  </section>

  <section class="controls">
    <input id="search" placeholder="Search libraries, descriptions, categories...">
    <select id="category">
      <option value="">All categories</option>
    </select>
    <select id="sort">
      <option value="name">A–Z</option>
      <option value="category">Category</option>
    </select>
  </section>

  <section id="grid" class="grid"></section>
  <div id="empty" class="empty">No libraries found. Try another search or category.</div>
</main>

<footer>
  Generated from README.md
</footer>

<script>
const items = $data;

const grid = document.getElementById('grid');
const search = document.getElementById('search');
const category = document.getElementById('category');
const sort = document.getElementById('sort');
const empty = document.getElementById('empty');

document.getElementById('total').textContent = items.length;

const categories = [...new Set(items.map(i => i.category))].sort();
document.getElementById('categories').textContent = categories.length;

categories.forEach(cat => {
  const opt = document.createElement('option');
  opt.value = cat;
  opt.textContent = cat;
  category.appendChild(opt);
});

function render() {
  const q = search.value.toLowerCase();
  const cat = category.value;

  let filtered = items.filter(item => {
    const haystack = [
      item.name,
      item.description,
      item.category
    ].join(' ').toLowerCase();

    return haystack.includes(q) && (!cat || item.category === cat);
  });

  filtered.sort((a, b) => {
    if (sort.value === 'category') {
      return a.category.localeCompare(b.category) || a.name.localeCompare(b.name);
    }
    return a.name.localeCompare(b.name);
  });

  const starsFormatter = new Intl.NumberFormat('en-US');

  grid.innerHTML = filtered.map(item => `
    <article class="card">
      <h2>
        <span class="badge">\${item.badge || ''}</span>
        <a href="\${item.url}" target="_blank" rel="noopener">\${escapeHtml(item.name)}</a>
      </h2>
      <p class="desc">\${escapeHtml(item.description)}</p>
      <div class="meta">
        <span class="tag">\${escapeHtml(item.category)}</span>
        <span class="tag">\${new URL(item.url).hostname.replace('www.', '')}</span>
        \${item.stars !== null ? `<span class="tag">⭐ \${starsFormatter.format(item.stars)}</span>` : ''}
      </div>
    </article>
  `).join('');

  empty.style.display = filtered.length ? 'none' : 'block';
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, s => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;'
  }[s]));
}

search.addEventListener('input', render);
category.addEventListener('change', render);
sort.addEventListener('change', render);

render();
</script>
</body>
</html>
HTML;

if (!is_dir(__DIR__ . '/docs')) {
    mkdir(__DIR__ . '/docs');
}

file_put_contents(__DIR__ . '/docs/index.html', $html);

echo "Generated docs/index.html with " . count($items) . " libraries\n";
