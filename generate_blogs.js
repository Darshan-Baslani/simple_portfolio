const fs = require('fs');
const path = require('path');

const ROOT_DIR = __dirname;
const BLOG_DIR = path.join(ROOT_DIR, 'blog');
const BLOGS_DATA_DIR = path.join(ROOT_DIR, 'blogs');

// Read manifest
const manifestPath = path.join(BLOGS_DATA_DIR, 'manifest.json');
const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));

// Create target directory if doesn't exist
if (!fs.existsSync(BLOG_DIR)) {
    fs.mkdirSync(BLOG_DIR, { recursive: true });
}

// 1. Generate blog/index.html (The main blogs tab)
// We will simply read index.html, grab everything except hero/about, and inject blogs HTML.
const indexHtmlTemplate = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Blogs - Darshan Baslani</title>
    <link rel="icon" type="image/png" href="/favicon.png">

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
    <header class="animate-enter">
        <div class="container">
            <nav>
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
        <section id="blogs" class="content-section animate-enter delay-4">
            <h2 class="section-title">Latest Thoughts</h2>
            <div class="narrative-text">
                <p>I write to clear my mind and share what I learn.</p>
            </div>
            
            <div class="grid">
                <!-- DYNAMIC CONTENT HERE -->
                <a href="/blog/08_the_tma_revolution/" class="card">
                    <span class="card-tag">CUDA</span>
                    <h3>The TMA Revolution (Async Copy)</h3>
                    <p>With the Hopper and Blackwell architectures, NVIDIA introduced the Tensor Memory Accelerator (TMA). Instead of having threads manually calculating pointers and copying data, a single thread can offload the entire tile copy to dedicated hardware.</p>
                </a>
                <a href="/blog/07_the_global_gemm/" class="card">
                    <span class="card-tag">CUDA</span>
                    <h3>The Global GEMM — Putting It All Together</h3>
                    <p>Writing a complete three-level tiled GEMM kernel from scratch using CuTe's TiledCopy, TiledMMA, and swizzled shared memory.</p>
                </a>
                <a href="/blog/06_hello_mma/" class="card">
                    <span class="card-tag">CUDA</span>
                    <h3>Hello, MMA — Your First Tensor Core Instruction</h3>
                    <p>How to use CuTe's TiledMMA to execute a matrix multiply-accumulate on NVIDIA Tensor Cores.</p>
                </a>
                <a href="/blog/05_swizzling/" class="card">
                    <span class="card-tag">CUDA</span>
                    <h3>Swizzling ; Avoiding Shared Memory Bank Conflicts</h3>
                    <p>How CuTe's Swizzle XORs address bits to eliminate shared memory bank conflicts with a single line of code.</p>
                </a>
                <a href="/blog/04_the_parallel_copy/" class="card">
                    <span class="card-tag">CUDA</span>
                    <h3>The Parallel Copy ; Orchestrating Threads with TiledCopy</h3>
                    <p>How TiledCopy bundles thread layout, copy atoms, and value layout into one declarative object for coordinated, vectorized parallel copies.</p>
                </a>
                <a href="/blog/03_the_naive_copy/" class="card">
                    <span class="card-tag">CUDA</span>
                    <h3>The Naive Copy ; Scalar vs. Vectorized Memory Movement</h3>
                    <p>Why scalar copies leave 75% of memory bandwidth on the table, and how CuTe's auto-vectorization fixes it.</p>
                </a>
                <a href="/blog/02_the_art_of_slicing/" class="card">
                    <span class="card-tag">CUDA</span>
                    <h3>The Art of Slicing ; Partitioning Data Across Blocks and Threads</h3>
                    <p>How CuTe's local_tile and local_partition replace manual index math to slice matrices across CTAs and threads.</p>
                </a>
                <a href="/blog/01_hello_layout/" class="card">
                    <span class="card-tag">CUDA</span>
                    <h3>Hello, Layout! ; Visualizing Memory in CuTe</h3>
                    <p>Understanding CuTe Layouts: how shape and stride turn flat memory into multidimensional grids.</p>
                </a>
                <a href="https://medium.com/@dcbaslani/beating-pytorch-writing-a-faster-softmax-kernel-in-cuda-0d0a237cda57" class="card" target="_blank">
                    <span class="card-tag">CUDA</span>
                    <h3>Beating PyTorch: Writing a Faster Softmax Kernel in CUDA</h3>
                    <p>Writing a faster Softmax kernel in CUDA than PyTorch's implementation.</p>
                </a>
                <a href="https://medium.com/@dcbaslani/stable-diffusion-1-5-how-i-optimized-it-a-worklog-09aa56498cf2" class="card" target="_blank">
                    <span class="card-tag">Machine Learning</span>
                    <h3>Stable Diffusion 1.5: How I Optimized It</h3>
                    <p>A detailed worklog on optimizing Stable Diffusion 1.5 for performance.</p>
                </a>
                <a href="https://medium.com/@dcbaslani/propositional-logic-25abd05e5aac" class="card" target="_blank">
                    <span class="card-tag">Logic</span>
                    <h3>Propositional Logic</h3>
                    <p>A deep dive into the fundamental building blocks of mathematical logic.</p>
                </a>
                <a href="https://medium.com/@dcbaslani/raw-dawgging-linear-regression-4a533e1f8ad2" class="card" target="_blank">
                    <span class="card-tag">Machine Learning</span>
                    <h3>Raw Dawgging Linear Regression</h3>
                    <p>Understanding Linear Regression by building it from the ground up.</p>
                </a>
            </div>
        </section>
    </main>

    <footer class="animate-enter delay-4">
        <div class="container" id="contact">
            <p>You can reach me out at</p>
            <div class="social-links">
                <a href="mailto:dcbaslani@gmail.com">Email</a>
                <a href="https://www.linkedin.com/in/darshan-baslani-7086051b6/">LinkedIn</a>
                <a href="https://twitter.com/neuronfitting">Twitter</a>
                <a href="https://github.com/Darshan-Baslani">GitHub</a>
            </div>
            <br>
            <p style="font-size: 0.8rem; opacity: 0.5;">&copy; 2026 Darshan.</p>
        </div>
    </footer>

    <script>
        const toggleButton = document.getElementById('theme-toggle');
        const htmlElement = document.documentElement;
        const currentTheme = localStorage.getItem('theme');
        if (currentTheme) {
            htmlElement.setAttribute('data-theme', currentTheme);
            if (currentTheme === 'dark') toggleButton.textContent = '☀️';
        }
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

// 2. Generate blog/[slug]/index.html for all posts
const blogPostTemplate = (slug, title) => `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${title} — Darshan Baslani</title>
    <link rel="icon" type="image/png" href="/favicon.png">

    <script async src="https://www.googletagmanager.com/gtag/js?id=G-7S3KXBK6D2"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag() { dataLayer.push(arguments); }
        gtag('js', new Date());
        gtag('config', 'G-7S3KXBK6D2');
    </script>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/cpp.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">

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

        .blog-loading, .blog-error { text-align: center; padding: 6rem 2rem; }
        .blog-loading .spinner { width: 32px; height: 32px; border: 3px solid var(--border-color); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; margin: 0 auto 1rem; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .blog-error h2 { font-size: 1.5rem; margin-bottom: 0.5rem; }
        .blog-error p { color: var(--text-muted); margin-bottom: 1.5rem; }
        .blog-error a { color: var(--accent); }
        .blog-error a:hover { text-decoration: underline; }

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
            <nav>
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
        <div id="blog-loading" class="blog-loading">
            <div class="spinner"></div>
            <p>Loading post...</p>
        </div>

        <div id="blog-error" class="blog-error" style="display: none;">
            <h2>Post not found</h2>
            <p>The blog post you're looking for doesn't exist or has been removed.</p>
            <a href="/blog/">&larr; Back to all posts</a>
        </div>

        <article id="blog-article" style="display: none;">
            <div class="blog-meta">
                <a href="/blog/" class="blog-back">&larr; Back to all posts</a>
                <div>
                    <span id="blog-tag" class="blog-tag"></span>
                    <span id="blog-date" class="blog-date"></span>
                </div>
            </div>
            <div id="blog-content" class="blog-content"></div>
        </article>
    </main>

    <footer>
        <div class="container" id="contact">
            <p>You can reach me out at</p>
            <div class="social-links">
                <a href="mailto:dcbaslani@gmail.com">Email</a>
                <a href="https://www.linkedin.com/in/darshan-baslani-7086051b6/">LinkedIn</a>
                <a href="https://twitter.com/neuronfitting">Twitter</a>
                <a href="https://github.com/Darshan-Baslani">GitHub</a>
            </div>
            <br>
            <p style="font-size: 0.8rem; opacity: 0.5;">&copy; 2026 Darshan.</p>
        </div>
    </footer>

    <script>
        const toggleButton = document.getElementById('theme-toggle');
        const htmlElement = document.documentElement;
        const currentTheme = localStorage.getItem('theme');
        if (currentTheme) {
            htmlElement.setAttribute('data-theme', currentTheme);
            if (currentTheme === 'dark') toggleButton.textContent = '☀️';
        }
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

        // Hardcoded slug to avoid parsing it from URL
        const SLUG = \`${slug}\`;

        (async function () {
            const loadingEl = document.getElementById('blog-loading');
            const errorEl = document.getElementById('blog-error');
            const articleEl = document.getElementById('blog-article');
            const contentEl = document.getElementById('blog-content');
            const tagEl = document.getElementById('blog-tag');
            const dateEl = document.getElementById('blog-date');

            function showError() {
                loadingEl.style.display = 'none';
                errorEl.style.display = 'block';
            }

            try {
                // Fetch manifest to get metadata
                const manifestRes = await fetch('/blogs/manifest.json');
                if (!manifestRes.ok) throw new Error('Manifest not found');
                const manifest = await manifestRes.json();

                const post = manifest.posts.find(p => p.slug === SLUG);
                if (!post) {
                    showError();
                    return;
                }

                // Fetch the markdown file
                const mdRes = await fetch(\`/blogs/\${SLUG}.md\`);
                if (!mdRes.ok) {
                    showError();
                    return;
                }
                const mdTextRaw = await mdRes.text();
                let mdText = mdTextRaw.replaceAll('—', '; ');
                
                // Rewrite relative markdown links to point to the new /blog/slug/ URLs
                mdText = mdText.replace(/\]\((?:\.\/|\.\.\/)?([a-zA-Z0-9_-]+)\.md\)/g, '](/blog/$1/)');

                // Configure marked
                marked.setOptions({ gfm: true, breaks: false });

                // Render markdown
                contentEl.innerHTML = marked.parse(mdText);
                contentEl.querySelectorAll('pre code').forEach(block => {
                    hljs.highlightElement(block);
                });

                // Fill metadata
                tagEl.textContent = post.tag;
                const dateObj = new Date(post.date + 'T00:00:00');
                dateEl.textContent = dateObj.toLocaleDateString('en-US', {
                    year: 'numeric',
                    month: 'long',
                    day: 'numeric'
                });

                // Show article
                loadingEl.style.display = 'none';
                articleEl.style.display = 'block';

            } catch (err) {
                console.error('Failed to load blog:', err);
                showError();
            }
        })();
    </script>
    <script src="/trackers.js"></script>
</body>
</html>`;

for (const post of manifest.posts) {
    const postDir = path.join(BLOG_DIR, post.slug);
    if (!fs.existsSync(postDir)) {
        fs.mkdirSync(postDir, { recursive: true });
    }
    const htmlPath = path.join(postDir, 'index.html');
    fs.writeFileSync(htmlPath, blogPostTemplate(post.slug, post.title.replaceAll('—', '; ')));
    console.log(`Generated blog/${post.slug}/index.html`);
}
