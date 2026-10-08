"""Build the deploy/ bundle from site/public.

- Content-hash all static assets and rewrite every reference (HTML hrefs AND
  the runtime fetch() inside app.js / gan.js).
- HASHING HAPPENS LAST for any file whose bytes are rewritten: app.js is
  repointed at the hashed model name BEFORE app.js itself is hashed, so the
  filename hash always matches the served bytes (immutable caching is safe).
- Copy comparison.json (runtime-fetched by summary.js).
- Inject a per-page Content-Security-Policy <meta> with sha256 hashes for the
  executable inline scripts (JSON-LD data blocks need none). Meta CSP is
  page-scoped so it sidesteps Vercel header-source path matching; keep
  X-Frame-Options in the header config because frame-ancestors is ignored
  in meta tags.
- Assert: no inline style= attributes (they'd break strict style-src),
  no stale unhashed refs, every referenced file exists on disk.
- Emit deploy/vercel.json with one literal cache rule per hashed file.
"""
import base64
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "site/public"
OUT = ROOT / "deploy"
# data/library assets: hashed directly (bytes never rewritten afterwards)
DATA_HASHED = ["model.js", "styles.css", "summary.js", "gan_results.json", "model.json", "favicon.svg"]
# entry scripts: repointed at hashed data names FIRST, then hashed
APP_HASHED = ["app.js", "gan.js"]
KEEP = ["index.html", "summary.html", "about.html", "how.html", "gan.html", "robots.txt",
        "404.html", "favicon.ico", "apple-touch-icon.png", "sitemap.xml", "og-image.png"]


def h8(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:8]


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    for name in KEEP:
        src = SRC / name
        if src.exists():
            shutil.copy(src, OUT / name)

    # keep + hash-list everything; vercel.json comes from site/ root
    vj = ROOT / "site/vercel.json"
    if vj.exists():
        shutil.copy(vj, OUT / "vercel.json")

    # copy + hash-list hashed assets (app/model/styles/summary js, model.json, favicon)
    cj = SRC / "comparison.json"
    if cj.exists():
        shutil.copy(cj, OUT / "comparison.json")
    for name in DATA_HASHED + APP_HASHED:
        src = SRC / name
        if src.exists():
            shutil.copy(src, OUT / name)

    mapping = {}

    # 1. hash data assets first (their bytes are final)
    for name in DATA_HASHED:
        src = OUT / name
        if not src.exists():
            print(f"  skip (missing): {name}")
            continue
        hashed = f"{src.stem}.{h8(src)}{src.suffix}"
        src.rename(OUT / hashed)
        mapping[name] = hashed
        print(f"  {name:14s} -> {hashed}")

    # 2. repoint runtime fetches inside the still-unhashed entry scripts
    if "model.json" in mapping:
        ap = OUT / "app.js"
        text = ap.read_text()
        new, n = re.subn(r'(["\'])/model\.json(["\'])', r"\g<1>/" + mapping["model.json"] + r"\g<2>", text)
        if not n:
            raise SystemExit("app.js does not reference /model.json as expected")
        ap.write_text(new)
        print(f"  repointed app.js fetch -> /{mapping['model.json']}")
    if "gan_results.json" in mapping:
        gp = OUT / "gan.js"
        text = gp.read_text()
        if '"/gan_results.json"' not in text:
            raise SystemExit("gan.js does not reference /gan_results.json as expected")
        gp.write_text(text.replace('"/gan_results.json"', '"/' + mapping["gan_results.json"] + '"'))
        print(f"  repointed gan.js fetch -> /{mapping['gan_results.json']}")

    # 3. hash the entry scripts LAST so the filename matches served bytes
    for name in APP_HASHED:
        src = OUT / name
        if not src.exists():
            print(f"  skip (missing): {name}")
            continue
        hashed = f"{src.stem}.{h8(src)}{src.suffix}"
        src.rename(OUT / hashed)
        mapping[name] = hashed
        print(f"  {name:14s} -> {hashed} (after repoint)")

    for page in OUT.glob("*.html"):
        text = page.read_text()
        before = text
        for original, hashed in mapping.items():
            text = re.sub(r'(?<=[\"\'])/' + re.escape(original) + r'(?=[\"\'])', "/" + hashed, text)
        if text != before:
            page.write_text(text)
            print(f"  rewrote refs in {page.name}")

    # summary.js fetches /comparison.json — unhashed data file, fine as-is.
    # Now CSP injection.
    def _csp_for(page_text):
        scripts, styles = [], []
        for m in re.finditer(r"<script(?![^>]*src=)([^>]*)>([\s\S]*?)</script>", page_text):
            attrs, body = m.group(1), m.group(2)
            if "application/ld+json" in attrs:
                continue
            if body.strip():
                d = hashlib.sha256(body.encode()).digest()
                scripts.append("'sha256-" + base64.b64encode(d).decode() + "'")
        for m in re.finditer(r"<style[^>]*>([\s\S]*?)</style>", page_text):
            if m.group(1).strip():
                d = hashlib.sha256(m.group(1).encode()).digest()
                styles.append("'sha256-" + base64.b64encode(d).decode() + "'")
        script_src = " ".join(["'self'", *scripts])
        style_src = " ".join(["'self'", *styles])
        return (
            "default-src 'self'; "
            "script-src " + script_src + "; "
            "style-src " + style_src + "; "
            "img-src 'self'; font-src 'self'; connect-src 'self'; "
            "object-src 'none'; base-uri 'self'"
        )

    for page in OUT.glob("*.html"):
        text = page.read_text()
        if 'http-equiv="Content-Security-Policy"' in text:
            raise SystemExit(page.name + " already has a CSP meta tag")
        if re.search(r"\sstyle=", text):
            raise SystemExit(page.name + " still has inline style attributes: CSP would break it")
        policy = _csp_for(text)
        tag = '<meta http-equiv="Content-Security-Policy" content="' + policy + '" />\n'
        new_text, n = re.subn(r'(<meta charset="utf-8" />\n)', r"\g<1>" + tag, text, count=1)
        if not n:
            raise SystemExit(page.name + ": no charset meta to anchor CSP injection")
        page.write_text(new_text)
        print(f"  CSP {page.name}: {policy[:80]}...")

    # vercel.json
    immutable_rules = [
        {"source": "/" + hashed,
         "headers": [{"key": "Cache-Control", "value": "public, max-age=31536000, immutable"}]}
        for hashed in mapping.values()
    ]
    cfg = {
        "$schema": "https://openapi.vercel.sh/vercel.json",
        "cleanUrls": True,
        "trailingSlash": False,
        "builds": [{"src": "**", "use": "@vercel/static"}],
        "headers": [
            *immutable_rules,
            {"source": "/comparison.json",
             "headers": [{"key": "Content-Type", "value": "application/json"},
                          {"key": "Cache-Control", "value": "public, max-age=0, must-revalidate"}]},
            {"source": "/(.*).html",
             "headers": [{"key": "Cache-Control", "value": "public, max-age=0, must-revalidate"}]},
            {"source": "/(.*)",
             "headers": [
                 {"key": "X-Content-Type-Options", "value": "nosniff"},
                 {"key": "Referrer-Policy", "value": "strict-origin-when-cross-origin"},
                 {"key": "X-Frame-Options", "value": "SAMEORIGIN"},
                 {"key": "Permissions-Policy", "value": "geolocation=(), microphone=(), camera=(), interest-cohort=()"},
             ]},
        ],
    }
    (OUT / "vercel.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"\nwrote {OUT}")

    # _headers for Cloudflare Pages (vercel.json cache rules are ignored there)
    hl = []
    for hashed in mapping.values():
        hl.append("/" + hashed + "\n  Cache-Control: public, max-age=31536000, immutable\n")
    for page in ["index.html", "summary.html", "about.html", "how.html", "gan.html", "404.html",
                 "index", "summary", "about", "how", "gan"]:
        hl.append("/" + page + "\n  Cache-Control: public, max-age=0, must-revalidate\n")
    for f in ["comparison.json", "sitemap.xml", "robots.txt"]:
        hl.append("/" + f + "\n  Cache-Control: public, max-age=3600\n")
    (OUT / "_headers").write_text("\n".join(hl) + "\n")
    print("  wrote _headers")

    # sanity gates
    stale = []
    for page in OUT.glob("*.html"):
        text = page.read_text()
        for name in DATA_HASHED + APP_HASHED:
            if f'"/{name}"' in text:
                stale.append(f"{page.name} -> /{name}")
    if stale:
        raise SystemExit("STALE REFERENCES:\n  " + "\n  ".join(stale))
    missing = []
    for page in OUT.glob("*.html"):
        for ref in re.findall(r'(?:href|src|content)="/([^"]+\.(?:js|css|json|svg|png|xml|txt))"', page.read_text()):
            if not (OUT / ref).exists():
                missing.append(f"{page.name} -> /{ref}")
    js = next(OUT.glob("app.*.js"))
    for ref in re.findall(r'["\'](/model\.[0-9a-f]{8}\.json)["\']', js.read_text()):
        if not (OUT / ref.lstrip("/")).exists():
            missing.append(f"app.js fetch -> {ref}")
    for ref in re.findall(r'["\'](/comparison\.json)["\']', (next(OUT.glob("summary.*.js"))).read_text()):
        if not (OUT / ref.lstrip("/")).exists():
            missing.append(f"summary.js -> {ref}")
    # filename hash must equal content hash for immutable entries
    for hashed in mapping.values():
        p = OUT / hashed
        stem, dot, suffix = hashed.rpartition(".")
        file_hash = stem.rsplit(".", 1)[-1]
        if file_hash != h8(p):
            missing.append(f"hash mismatch: {hashed} content is {h8(p)}")
    if missing:
        raise SystemExit("MISSING ASSETS:\n  " + "\n  ".join(sorted(set(missing))))
    print("all references resolve; hashes match bytes; inline styles absent; CSP present on every page")


if __name__ == "__main__":
    main()
