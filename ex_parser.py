from __future__ import annotations

import html
import re


def view_base(view: list[str]) -> str:
    if not isinstance(view, list) or len(view) < 2:
        return "https://exhentai.org/"
    return f"https://exhentai.org/s/{str(view[0])}/{str(view[1])}"


def normalize_img_url(url: str) -> str:
    u = html.unescape(str(url or "").strip())
    if not u:
        return ""
    if u.startswith("//"):
        return "https:" + u
    if u.startswith("/"):
        return "https://exhentai.org" + u
    return u


def is_509_marker_url(url: str) -> bool:
    base = str(url or "").strip().lower().split("?", 1)[0]
    return base.endswith("/509.gif") or base.endswith("/509s.gif")


def extract_img_src(text: str) -> str:
    t = str(text or "")
    m = re.search(r'<img[^>]*id=["\']img["\'][^>]*src=["\']([^"\']+)', t, re.IGNORECASE)
    if not m:
        m = re.search(r'<img[^>]*src=["\']([^"\']+)["\'][^>]*id=["\']img["\']', t, re.IGNORECASE)
    if not m:
        m = re.search(r'<img[^>]*src=["\']([^"\']+)["\'][^>]*style', t, re.IGNORECASE)
    return normalize_img_url(m.group(1) if m else "")


def extract_nl_key(text: str) -> str:
    t = str(text or "")
    m = re.search(r'onclick=["\']return\s+nl\(["\']([^"\']+)["\']\)', t, re.IGNORECASE)
    if not m:
        m = re.search(r'nl\(["\']([^"\']+)["\']\)', t, re.IGNORECASE)
    return html.unescape(m.group(1).strip()) if m else ""


def extract_show_key(text: str) -> str:
    t = str(text or "")
    for pat in [
        r'showkey\s*=\s*["\']([^"\']+)["\']',
        r'"showkey"\s*:\s*"([^"]+)"',
        r"'showkey'\s*:\s*'([^']+)'",
    ]:
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            return html.unescape(m.group(1).strip())
    return ""


def extract_origin_url(text: str) -> str:
    t = str(text or "")
    m = re.search(r'<a[^>]*href=["\']([^"\']*fullimg[^"\']*)["\']', t, re.IGNORECASE)
    if not m:
        m = re.search(r"prompt\('Copy the URL below\.',\s*'([^']+)'\)", t, re.IGNORECASE)
    if not m:
        m = re.search(r'<a[^>]*href=["\']([^"\']+)["\'][^>]*>\s*(?:Download original|Original image)', t, re.IGNORECASE)
    return normalize_img_url(m.group(1) if m else "")


def split_gid_page(page_id: str) -> tuple[str, int | None]:
    s = str(page_id or "").strip()
    if not s:
        return "", None
    if "-" in s:
        gid, pg = s.split("-", 1)
        if gid.isdigit() and pg.isdigit():
            return gid, int(pg)
    if s.isdigit():
        return s, 0
    return "", None
