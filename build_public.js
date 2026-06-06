const fs = require('fs');
const path = require('path');

const ROOT_DIR = __dirname;
const OUT_DIR = path.join(ROOT_DIR, 'public');

const FILES = [
    'blog.html',
    'icon.webp',
    'index.html',
    'llms-full.txt',
    'llms.txt',
    'robots.txt',
    'sitemap.xml',
    'trackers.js',
];

const DIRS = [
    'blog',
    'docs',
];

fs.rmSync(OUT_DIR, { recursive: true, force: true });
fs.mkdirSync(OUT_DIR, { recursive: true });

for (const file of FILES) {
    const src = path.join(ROOT_DIR, file);
    if (fs.existsSync(src)) {
        fs.copyFileSync(src, path.join(OUT_DIR, file));
    }
}

for (const dir of DIRS) {
    const src = path.join(ROOT_DIR, dir);
    if (fs.existsSync(src)) {
        fs.cpSync(src, path.join(OUT_DIR, dir), { recursive: true });
    }
}

console.log(`Created ${path.relative(ROOT_DIR, OUT_DIR)} deploy output`);
