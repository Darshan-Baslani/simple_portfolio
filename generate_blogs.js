const fs = require('fs');
const path = require('path');
const { marked } = require('marked');

const ROOT_DIR = __dirname;
const BLOG_DIR = path.join(ROOT_DIR, 'blog');
const BLOGS_DATA_DIR = path.join(ROOT_DIR, 'blogs');
const SITE_URL = 'https://darshanbaslani.com';

// Read manifest
const manifestPath = path.join(BLOGS_DATA_DIR, 'manifest.json');
const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));

// Create target directory if doesn't exist
if (!fs.existsSync(BLOG_DIR)) {
    fs.mkdirSync(BLOG_DIR, { recursive: true });
}

// Configure marked for SSR
marked.setOptions({ gfm: true, breaks: false });

// Helper: format date for display
function formatDate(dateStr) {
    const d = new Date(dateStr + 'T00:00:00');
    return d.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
}

// Helper: format date for ISO (structured data)
function isoDate(dateStr) {
    return dateStr; // Already in YYYY-MM-DD
}

// Helper: estimate reading time
function readingTime(text) {
    const words = text.split(/\s+/).length;
    return Math.max(1, Math.ceil(words / 200));
}

// Helper: strip markdown to plain text (for meta descriptions)
function stripMarkdown(md) {
    return md
        .replace(/```[\s\S]*?```/g, '')
        .replace(/`[^`]+`/g, '')
        .replace(/!\[.*?\]\(.*?\)/g, '')
        .replace(/\[([^\]]+)\]\(.*?\)/g, '$1')
        .replace(/#{1,6}\s/g, '')
        .replace(/[*_~|>-]/g, '')
        .replace(/\n+/g, ' ')
        .trim()
        .substring(0, 300);
}

// Helper: escape HTML in JSON-LD strings
function escJsonLd(str) {
    return str.replace(/"/g, '\\"').replace(/\n/g, '\\n');
}

// ============================================================
// 1. Generate blog/index.html (The main blogs listing page)
// ============================================================
const blogCards = manifest.posts.slice().reverse().map(post => {
    const href = post.slug.startsWith('http') ? post.slug : `/blog/${post.slug}/`;
    const isExternal = post.slug.startsWith('http');
    const target = isExternal ? ' target="_blank" rel="noopener"' : '';
    return `                <a href="${href}" class="card"${target}>
                    <span class="card-tag">${post.tag}</span>
                    <h3>${post.title.replaceAll('—', '; ')}</h3>
                    <p>${post.description}</p>
                    <time datetime="${post.date}" class="card-date">${formatDate(post.date)}</time>
                </a>`;
}).join('\n');

// Add the external Medium posts that aren't in manifest
const externalPosts = `                <a href="https://medium.com/@dcbaslani/my-2-cents-on-doing-hard-things-9af575ae867b" class="card" target="_blank" rel="noopener">
                    <span class="card-tag">Life</span>
                    <h3>My 2 Cents on Doing Hard Things</h3>
                    <p>Reflections on why hard things are worth doing and how to keep going when it gets tough.</p>
                </a>
                <a href="https://medium.com/@dcbaslani/beating-pytorch-writing-a-faster-softmax-kernel-in-cuda-0d0a237cda57" class="card" target="_blank" rel="noopener">
                    <span class="card-tag">CUDA</span>
                    <h3>Beating PyTorch: Writing a Faster Softmax Kernel in CUDA</h3>
                    <p>Writing a faster Softmax kernel in CUDA than PyTorch's implementation.</p>
                </a>
                <a href="https://medium.com/@dcbaslani/stable-diffusion-1-5-how-i-optimized-it-a-worklog-09aa56498cf2" class="card" target="_blank" rel="noopener">
                    <span class="card-tag">Machine Learning</span>
                    <h3>Stable Diffusion 1.5: How I Optimized It</h3>
                    <p>A detailed worklog on optimizing Stable Diffusion 1.5 for performance.</p>
                </a>
                <a href="https://medium.com/@dcbaslani/propositional-logic-25abd05e5aac" class="card" target="_blank" rel="noopener">
                    <span class="card-tag">Logic</span>
                    <h3>Propositional Logic</h3>
                    <p>A deep dive into the fundamental building blocks of mathematical logic.</p>
                </a>
                <a href="https://medium.com/@dcbaslani/raw-dawgging-linear-regression-4a533e1f8ad2" class="card" target="_blank" rel="noopener">
                    <span class="card-tag">Machine Learning</span>
                    <h3>Raw Dawgging Linear Regression</h3>
                    <p>Understanding Linear Regression by building it from the ground up.</p>
                </a>`;

// JSON-LD for the blog listing (CollectionPage)
const blogListJsonLd = JSON.stringify({
    "@context": "https://schema.org",
    "@type": "CollectionPage",
    "name": "Blog — Darshan Baslani",
    "description": "Technical blog posts on CUDA programming, GPU kernel optimization, CuTe library, and machine learning systems by Darshan Baslani.",
    "url": `${SITE_URL}/blog/`,
    "author": {
        "@type": "Person",
        "name": "Darshan Baslani",
        "url": SITE_URL
    },
    "mainEntity": {
        "@type": "ItemList",
        "itemListElement": manifest.posts.map((post, i) => ({
            "@type": "ListItem",
            "position": i + 1,
            "url": `${SITE_URL}/blog/${post.slug}/`,
            "name": post.title
        }))
    }
}, null, 2);

const indexHtmlTemplate = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Blog — Darshan Baslani | CUDA, GPU Programming & ML Systems</title>
    <meta name="description" content="Technical blog posts on CUDA programming, GPU kernel optimization, CuTe library, and machine learning systems by Darshan Baslani.">
    <meta name="author" content="Darshan Baslani">
    <link rel="canonical" href="${SITE_URL}/blog/">
    <link rel="icon" type="image/webp" href="/icon.webp">

    <!-- Early theme application to prevent FOUC -->
    <script>
        (function() {
            var t = localStorage.getItem('theme');
            if (t) document.documentElement.setAttribute('data-theme', t);
        })();
    </script>

    <!-- Open Graph -->
    <meta property="og:type" content="website">
    <meta property="og:title" content="Blog — Darshan Baslani">
    <meta property="og:description" content="Technical blog posts on CUDA programming, GPU kernel optimization, CuTe library, and machine learning systems.">
    <meta property="og:url" content="${SITE_URL}/blog/">
    <meta property="og:site_name" content="Darshan Baslani">

    <!-- Twitter Card -->
    <meta name="twitter:card" content="summary">
    <meta name="twitter:site" content="@neuronfitting">
    <meta name="twitter:title" content="Blog — Darshan Baslani">
    <meta name="twitter:description" content="Technical blog posts on CUDA programming, GPU kernel optimization, CuTe library, and machine learning systems.">

    <!-- JSON-LD Structured Data -->
    <script type="application/ld+json">
${blogListJsonLd}
    </script>

    <script async src="https://www.googletagmanager.com/gtag/js?id=G-7S3KXBK6D2"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag() { dataLayer.push(arguments); }
        gtag('js', new Date());
        gtag('config', 'G-7S3KXBK6D2');
    </script>

    <style>
        :root {
            --bg-color: #fcfbf9;
            --text-main: #2d2d2d;
            --text-muted: #6e6e6e;
            --accent: #d4a373;
            --highlight: #faedcd;
            --font-main: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            --card-bg: #ffffff;
            --shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.05);
            --tag-bg: #f0f0f0;
            --tag-text: #666;
            --border-color: #eee;
        }

        [data-theme="dark"] {
            --bg-color: #121212;
            --text-main: #e0e0e0;
            --text-muted: #b0b0b0;
            --card-bg: #1e1e1e;
            --shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
            --highlight: #b08968;
            --tag-bg: #333;
            --tag-text: #ccc;
            --border-color: #333;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }
        html { scroll-behavior: smooth; }
        body {
            background-color: var(--bg-color);
            font-family: var(--font-main);
            color: var(--text-main);
            line-height: 1.6;
            overflow-x: hidden;
            transition: background-color 0.4s ease, color 0.4s ease;
        }
        a { text-decoration: none; color: inherit; transition: color 0.3s ease; }
        ul { list-style: none; }
        .container { max-width: 900px; margin: 0 auto; padding: 0 2rem; }
        
        nav { padding: 2rem 0; display: flex; justify-content: space-between; align-items: center; }
        .logo { font-weight: 700; font-size: 1.2rem; letter-spacing: -0.5px; }
        .nav-links { display: flex; gap: 2rem; align-items: center; }
        .nav-links a { font-size: 0.95rem; color: var(--text-muted); position: relative; transition: color 0.3s ease; }
        .nav-links a::after { content: ''; position: absolute; width: 0; height: 1px; bottom: -4px; left: 0; background-color: var(--accent); transition: width 0.3s ease; }
        .nav-links a:hover { color: var(--text-main); }
        .nav-links a:hover::after { width: 100%; }
        #theme-toggle { background: none; border: none; cursor: pointer; color: var(--text-muted); font-size: 1.1rem; padding: 0.3rem; border-radius: 6px; transition: color 0.3s ease, background-color 0.3s ease; line-height: 1; }
        #theme-toggle:hover { color: var(--text-main); background-color: var(--tag-bg); }

        .section-title { font-size: 0.9rem; text-transform: uppercase; letter-spacing: 2px; color: var(--text-muted); margin-bottom: 2rem; border-left: 3px solid var(--accent); padding-left: 10px; }
        .content-section { margin-bottom: 8rem; margin-top: 3rem; }
        .narrative-text p { font-size: 1.1rem; margin-bottom: 1.5rem; color: var(--text-main); }

        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 2rem; }
        .card { background: var(--card-bg); padding: 2rem; border-radius: 12px; box-shadow: var(--shadow); border: 1px solid var(--border-color); transition: transform 0.35s cubic-bezier(0.2, 0.8, 0.2, 1), box-shadow 0.35s cubic-bezier(0.2, 0.8, 0.2, 1), border-color 0.35s ease, background-color 0.4s ease; cursor: pointer; }
        .card:hover { transform: translateY(-3px); box-shadow: 0 12px 32px -8px rgba(0, 0, 0, 0.08); border-color: var(--accent); }
        .card h3 { font-size: 1.3rem; margin-bottom: 0.5rem; }
        .card p { color: var(--text-muted); font-size: 0.95rem; }
        .card-tag { display: inline-block; font-size: 0.75rem; background-color: var(--tag-bg); padding: 4px 10px; border-radius: 6px; margin-bottom: 1rem; color: var(--tag-text); letter-spacing: 0.5px; font-weight: 500; transition: background-color 0.4s ease, color 0.4s ease; }
        .card-date { display: block; font-size: 0.8rem; color: var(--text-muted); margin-top: 0.75rem; }

        footer { padding: 4rem 0; border-top: 1px solid var(--border-color); text-align: center; color: var(--text-muted); font-size: 0.9rem; transition: border-color 0.4s ease; }
        .social-links { margin-top: 1rem; display: flex; justify-content: center; gap: 1.5rem; }
        .social-links a { font-weight: 500; transition: color 0.3s ease; }
        .social-links a:hover { color: var(--accent); }

        @media (max-width: 768px) {
            .nav-links { display: none; }
        }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <nav aria-label="Main navigation">
                <div class="logo"><a href="/">Darshan Baslani</a></div>
                <div class="nav-links">
                    <a href="/#about">About</a>
                    <a href="/blog/">Blogs</a>
                    <a href="/#contact">Contact</a>
                    <button id="theme-toggle" aria-label="Toggle Dark Mode">🌙</button>
                </div>
            </nav>
        </div>
    </header>

    <main class="container">
        <section id="blogs" class="content-section animate-enter delay-4" aria-label="Blog posts">
            <h1 class="section-title">Latest Thoughts</h1>
            <div class="narrative-text">
                <p>I write to clear my mind and share what I learn.</p>
            </div>
            
            <div class="grid">
${blogCards}
${externalPosts}
            </div>
        </section>
    </main>

    <footer role="contentinfo">
        <div class="container" id="contact">
            <p>You can reach me out at</p>
            <nav aria-label="Social links" class="social-links">
                <a href="mailto:dcbaslani@gmail.com" rel="me">Email</a>
                <a href="https://www.linkedin.com/in/darshan-baslani-7086051b6/" rel="me noopener noreferrer" target="_blank">LinkedIn</a>
                <a href="https://twitter.com/neuronfitting" rel="me noopener noreferrer" target="_blank">Twitter</a>
                <a href="https://github.com/Darshan-Baslani" rel="me noopener noreferrer" target="_blank">GitHub</a>
            </nav>
            <br>
            <p style="font-size: 0.8rem; opacity: 0.5;">&copy; 2026 Darshan.</p>
        </div>
    </footer>

    <script>
        const toggleButton = document.getElementById('theme-toggle');
        const htmlElement = document.documentElement;
        if (htmlElement.getAttribute('data-theme') === 'dark') toggleButton.textContent = '☀️';
        toggleButton.addEventListener('click', () => {
            if (htmlElement.getAttribute('data-theme') === 'dark') {
                htmlElement.setAttribute('data-theme', 'light');
                localStorage.setItem('theme', 'light');
                toggleButton.textContent = '🌙';
            } else {
                htmlElement.setAttribute('data-theme', 'dark');
                localStorage.setItem('theme', 'dark');
                toggleButton.textContent = '☀️';
            }
        });
    </script>
    <script src="/trackers.js"></script>
</body>
</html>`;

fs.writeFileSync(path.join(BLOG_DIR, 'index.html'), indexHtmlTemplate);
console.log('Created blog/index.html');

// ============================================================
// 2. Generate blog/[slug]/index.html for all posts (with SSR!)
// ============================================================
const blogPostTemplate = (slug, post, renderedHtml, readMin) => {
    const title = post.title.replaceAll('—', '; ');
    const postUrl = `${SITE_URL}/blog/${slug}/`;
    const dateFormatted = formatDate(post.date);

    // JSON-LD for individual blog post
    const postJsonLd = JSON.stringify({
        "@context": "https://schema.org",
        "@type": "BlogPosting",
        "headline": post.title,
        "description": post.description,
        "datePublished": isoDate(post.date),
        "dateModified": isoDate(post.date),
        "author": {
            "@type": "Person",
            "name": "Darshan Baslani",
            "url": SITE_URL
        },
        "publisher": {
            "@type": "Person",
            "name": "Darshan Baslani",
            "url": SITE_URL
        },
        "mainEntityOfPage": {
            "@type": "WebPage",
            "@id": postUrl
        },
        "keywords": post.tag,
        "wordCount": renderedHtml.split(/\s+/).length,
        "timeRequired": `PT${readMin}M`
    }, null, 2);

    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${title} — Darshan Baslani</title>
    <meta name="description" content="${post.description}">
    <meta name="author" content="Darshan Baslani">
    <link rel="canonical" href="${postUrl}">
    <link rel="icon" type="image/webp" href="/icon.webp">

    <!-- Early theme application to prevent FOUC -->
    <script>
        (function() {
            var t = localStorage.getItem('theme');
            if (t) document.documentElement.setAttribute('data-theme', t);
        })();
    </script>

    <!-- Open Graph -->
    <meta property="og:type" content="article">
    <meta property="og:title" content="${title}">
    <meta property="og:description" content="${post.description}">
    <meta property="og:url" content="${postUrl}">
    <meta property="og:site_name" content="Darshan Baslani">
    <meta property="article:published_time" content="${isoDate(post.date)}">
    <meta property="article:author" content="Darshan Baslani">
    <meta property="article:tag" content="${post.tag}">

    <!-- Twitter Card -->
    <meta name="twitter:card" content="summary">
    <meta name="twitter:site" content="@neuronfitting">
    <meta name="twitter:title" content="${title}">
    <meta name="twitter:description" content="${post.description}">

    <!-- JSON-LD Structured Data -->
    <script type="application/ld+json">
${postJsonLd}
    </script>

    <script async src="https://www.googletagmanager.com/gtag/js?id=G-7S3KXBK6D2"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag() { dataLayer.push(arguments); }
        gtag('js', new Date());
        gtag('config', 'G-7S3KXBK6D2');
    </script>

    <!-- Async-load highlight.js CSS to avoid render-blocking -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css" media="print" onload="this.media='all'">
    <noscript><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css"></noscript>

    <style>
        :root {
            --bg-color: #fcfbf9;
            --text-main: #2d2d2d;
            --text-muted: #6e6e6e;
            --accent: #d4a373;
            --highlight: #faedcd;
            --font-main: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            --card-bg: #ffffff;
            --shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.05);
            --tag-bg: #f0f0f0;
            --tag-text: #666;
            --border-color: #eee;
        }

        [data-theme="dark"] {
            --bg-color: #121212;
            --text-main: #e0e0e0;
            --text-muted: #b0b0b0;
            --card-bg: #1e1e1e;
            --shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
            --highlight: #b08968;
            --tag-bg: #333;
            --tag-text: #ccc;
            --border-color: #333;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }
        html { scroll-behavior: smooth; }
        body {
            background-color: var(--bg-color);
            font-family: var(--font-main);
            color: var(--text-main);
            line-height: 1.6;
            overflow-x: hidden;
            transition: background-color 0.4s ease, color 0.4s ease;
        }
        a { text-decoration: none; color: inherit; transition: color 0.3s ease; }
        .container { max-width: 900px; margin: 0 auto; padding: 0 2rem; }
        
        nav { padding: 2rem 0; display: flex; justify-content: space-between; align-items: center; }
        .logo { font-weight: 700; font-size: 1.2rem; letter-spacing: -0.5px; }
        .nav-links { display: flex; gap: 2rem; align-items: center; }
        .nav-links a { font-size: 0.95rem; color: var(--text-muted); position: relative; transition: color 0.3s ease; }
        .nav-links a::after { content: ''; position: absolute; width: 0; height: 1px; bottom: -4px; left: 0; background-color: var(--accent); transition: width 0.3s ease; }
        .nav-links a:hover { color: var(--text-main); }
        .nav-links a:hover::after { width: 100%; }
        #theme-toggle { background: none; border: none; cursor: pointer; color: var(--text-muted); font-size: 1.1rem; padding: 0.3rem; border-radius: 6px; transition: color 0.3s ease, background-color 0.3s ease; line-height: 1; }
        #theme-toggle:hover { color: var(--text-main); background-color: var(--tag-bg); }

        footer { padding: 4rem 0; border-top: 1px solid var(--border-color); text-align: center; color: var(--text-muted); font-size: 0.9rem; transition: border-color 0.4s ease; }
        .social-links { margin-top: 1rem; display: flex; justify-content: center; gap: 1.5rem; }
        .social-links a { font-weight: 500; transition: color 0.3s ease; }
        .social-links a:hover { color: var(--accent); }

        .blog-meta { margin-top: 2rem; margin-bottom: 2rem; }
        .blog-back { display: inline-block; color: var(--accent); font-size: 0.95rem; margin-bottom: 1.5rem; transition: opacity 0.3s ease; }
        .blog-back:hover { opacity: 0.7; }
        .blog-tag { display: inline-block; font-size: 0.75rem; background-color: var(--tag-bg); padding: 4px 8px; border-radius: 4px; color: var(--tag-text); margin-right: 0.75rem; }
        .blog-date { font-size: 0.85rem; color: var(--text-muted); }
        .blog-readtime { font-size: 0.85rem; color: var(--text-muted); margin-left: 0.5rem; }

        .blog-content { margin-bottom: 4rem; }
        .blog-content h1 { font-size: 2.2rem; line-height: 1.2; font-weight: 800; margin-bottom: 1.5rem; letter-spacing: -0.5px; }
        .blog-content h2 { font-size: 1.6rem; font-weight: 700; margin-top: 2.5rem; margin-bottom: 1rem; border-bottom: 1px solid var(--border-color); padding-bottom: 0.5rem; }
        .blog-content h3 { font-size: 1.25rem; font-weight: 600; margin-top: 2rem; margin-bottom: 0.75rem; }
        .blog-content p { font-size: 1.05rem; margin-bottom: 1.25rem; color: var(--text-main); }
        .blog-content strong { font-weight: 700; }
        .blog-content a { color: var(--accent); text-decoration: underline; }
        .blog-content a:hover { opacity: 0.8; }
        .blog-content ul, .blog-content ol { list-style: revert; margin-bottom: 1.25rem; padding-left: 2rem; }
        .blog-content li { margin-bottom: 0.4rem; font-size: 1.05rem; }
        .blog-content blockquote { border-left: 3px solid var(--accent); padding: 0.75rem 1.25rem; margin: 1.5rem 0; background: var(--card-bg); border-radius: 0 8px 8px 0; }
        .blog-content blockquote p { color: var(--text-muted); margin-bottom: 0; }
        .blog-content code { font-family: 'SF Mono', 'Fira Code', 'Fira Mono', Menlo, Consolas, monospace; font-size: 0.9em; background: var(--tag-bg); padding: 2px 6px; border-radius: 4px; }
        .blog-content pre { margin: 1.5rem 0; border-radius: 8px; overflow-x: auto; }
        .blog-content pre code { display: block; padding: 1.25rem; background: #0d1117; color: #e6edf3; font-size: 0.85rem; line-height: 1.6; border-radius: 8px; overflow-x: auto; }
        .blog-content table { width: 100%; border-collapse: collapse; margin: 1.5rem 0; font-size: 0.95rem; overflow-x: auto; display: block; }
        .blog-content th, .blog-content td { border: 1px solid var(--border-color); padding: 0.6rem 1rem; text-align: left; }
        .blog-content th { background: var(--card-bg); font-weight: 600; }
        .blog-content tr:nth-child(even) { background: var(--card-bg); }
        .blog-content hr { border: none; border-top: 1px solid var(--border-color); margin: 2.5rem 0; }
        .blog-content img { max-width: 100%; height: auto; border-radius: 8px; margin: 1.5rem 0; }

        @media (max-width: 768px) {
            .blog-content h1 { font-size: 1.7rem; }
            .blog-content h2 { font-size: 1.3rem; }
            .nav-links { display: none; }
            .blog-content pre code { font-size: 0.8rem; }
        }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <nav aria-label="Main navigation">
                <div class="logo"><a href="/">Darshan Baslani</a></div>
                <div class="nav-links">
                    <a href="/#about">About</a>
                    <a href="/blog/">Blogs</a>
                    <a href="/#contact">Contact</a>
                    <button id="theme-toggle" aria-label="Toggle Dark Mode">🌙</button>
                </div>
            </nav>
        </div>
    </header>

    <main class="container">
        <article itemscope itemtype="https://schema.org/BlogPosting">
            <div class="blog-meta">
                <a href="/blog/" class="blog-back">&larr; Back to all posts</a>
                <div>
                    <span class="blog-tag" itemprop="keywords">${post.tag}</span>
                    <time class="blog-date" datetime="${post.date}" itemprop="datePublished">${dateFormatted}</time>
                    <span class="blog-readtime">&middot; ${readMin} min read</span>
                </div>
            </div>
            <div id="blog-content" class="blog-content" itemprop="articleBody">
${renderedHtml}
            </div>
        </article>
    </main>

    <footer role="contentinfo">
        <div class="container" id="contact">
            <p>You can reach me out at</p>
            <nav aria-label="Social links" class="social-links">
                <a href="mailto:dcbaslani@gmail.com" rel="me">Email</a>
                <a href="https://www.linkedin.com/in/darshan-baslani-7086051b6/" rel="me noopener noreferrer" target="_blank">LinkedIn</a>
                <a href="https://twitter.com/neuronfitting" rel="me noopener noreferrer" target="_blank">Twitter</a>
                <a href="https://github.com/Darshan-Baslani" rel="me noopener noreferrer" target="_blank">GitHub</a>
            </nav>
            <br>
            <p style="font-size: 0.8rem; opacity: 0.5;">&copy; 2026 Darshan.</p>
        </div>
    </footer>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/cpp.min.js"></script>
    <script>
        // Theme toggle — icon init based on already-applied theme (set in <head>)
        const toggleButton = document.getElementById('theme-toggle');
        const htmlElement = document.documentElement;
        if (htmlElement.getAttribute('data-theme') === 'dark') toggleButton.textContent = '☀️';
        toggleButton.addEventListener('click', () => {
            if (htmlElement.getAttribute('data-theme') === 'dark') {
                htmlElement.setAttribute('data-theme', 'light');
                localStorage.setItem('theme', 'light');
                toggleButton.textContent = '🌙';
            } else {
                htmlElement.setAttribute('data-theme', 'dark');
                localStorage.setItem('theme', 'dark');
                toggleButton.textContent = '☀️';
            }
        });

        // Syntax highlighting for pre-rendered code blocks
        document.querySelectorAll('pre code').forEach(block => {
            hljs.highlightElement(block);
        });
    </script>
    <script src="/trackers.js"></script>
</body>
</html>`;
};

for (const post of manifest.posts) {
    const postDir = path.join(BLOG_DIR, post.slug);
    if (!fs.existsSync(postDir)) {
        fs.mkdirSync(postDir, { recursive: true });
    }

    // SSR: Read markdown, render to HTML at build time
    const mdPath = path.join(BLOGS_DATA_DIR, `${post.slug}.md`);
    let mdText = fs.readFileSync(mdPath, 'utf8');
    mdText = mdText.replaceAll('—', '; ');
    // Rewrite relative markdown links to point to the new /blog/slug/ URLs
    mdText = mdText.replace(/\]\((?:\.\/|\.\.\/)?([a-zA-Z0-9_-]+)\.md\)/g, '](/blog/$1/)');

    const renderedHtml = marked.parse(mdText);
    const readMin = readingTime(mdText);

    const htmlPath = path.join(postDir, 'index.html');
    fs.writeFileSync(htmlPath, blogPostTemplate(post.slug, post, renderedHtml, readMin));
    console.log(`Generated blog/${post.slug}/index.html (SSR, ${readMin} min read)`);
}

// ============================================================
// 3. Generate sitemap.xml
// ============================================================
const today = new Date().toISOString().split('T')[0];

let sitemapUrls = `    <url>
        <loc>${SITE_URL}/</loc>
        <lastmod>${today}</lastmod>
        <changefreq>monthly</changefreq>
        <priority>1.0</priority>
    </url>
    <url>
        <loc>${SITE_URL}/blog/</loc>
        <lastmod>${today}</lastmod>
        <changefreq>weekly</changefreq>
        <priority>0.9</priority>
    </url>`;

for (const post of manifest.posts) {
    sitemapUrls += `
    <url>
        <loc>${SITE_URL}/blog/${post.slug}/</loc>
        <lastmod>${post.date}</lastmod>
        <changefreq>monthly</changefreq>
        <priority>0.8</priority>
    </url>`;
}

const sitemap = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${sitemapUrls}
</urlset>
`;

fs.writeFileSync(path.join(ROOT_DIR, 'sitemap.xml'), sitemap);
console.log('Generated sitemap.xml');

console.log('\n✅ All blog pages generated with SSR + SEO meta tags + sitemap');
