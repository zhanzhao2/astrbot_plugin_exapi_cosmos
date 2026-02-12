from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import socket
import tempfile
import time
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import quote, urlsplit

import aiohttp

import ex_parser

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"


class ExDownloader:
    def __init__(self, get_proxy: Callable[[], str | None], get_cookie_header: Callable[[], str], call_bridge: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]], get_temp_root: Callable[[], Path], logger: Any):
        self._get_proxy = get_proxy
        self._get_cookie_header = get_cookie_header
        self._call_bridge = call_bridge
        self._get_temp_root = get_temp_root
        self._logger = logger

    async def download_image_via_gallery_dl(
        self,
        url: str,
        out_path: Path,
        headers: dict[str, str],
        proxy: str | None = None,
        fast: bool = False,
    ) -> str | None:
        gdl_dir = out_path.parent / f'.gdl_{out_path.stem}'
        gdl_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            'gallery-dl', '--no-skip', '--no-mtime', '-4', '--no-check-certificate',
            '-R', '2', '--http-timeout', ('25' if fast else '45'),
            '-a', headers.get('User-Agent', 'Mozilla/5.0'),
            '-D', str(gdl_dir), '-f', '/O',
        ]
        referer = headers.get('Referer', 'https://exhentai.org/')
        cmd += ['-o', f'extractor.*.headers.Referer={referer}']
        if proxy:
            cmd += ['--proxy', str(proxy)]
        cmd.append(url)
        try:
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            await proc.communicate()
            if proc.returncode != 0:
                return None
            files = [p for p in gdl_dir.rglob('*') if p.is_file() and p.stat().st_size > 1024 and not p.name.endswith('.part')]
            if not files:
                return None
            src = max(files, key=lambda p: p.stat().st_mtime)
            target = out_path.with_suffix(src.suffix or out_path.suffix)
            try:
                if target.exists():
                    target.unlink()
            except Exception:
                pass
            shutil.move(str(src), str(target))
            return str(target)
        except Exception:
            return None
        finally:
            try:
                shutil.rmtree(gdl_dir, ignore_errors=True)
            except Exception:
                pass

    async def download_image(
        self,
        url: str,
        temp_dir: Path | None = None,
        referer: str | None = None,
        session: aiohttp.ClientSession | None = None,
        host_fail: dict[str, int] | None = None,
        fast: bool = False,
        force_proxy: bool = False,
    ) -> str | None:
        """下载图片到本地临时目录，返回本地路径；失败返回 None。"""
        if not isinstance(url, str) or not url.strip():
            return None

        url = url.strip()
        if url.startswith("//"):
            url = "https:" + url
        if not url.startswith("http"):
            return None
        if ex_parser.is_509_marker_url(url):
            return None

        if temp_dir is None:
            temp_dir = self._get_temp_root() / "temp_img"
        temp_dir.mkdir(parents=True, exist_ok=True)

        ext = ".jpg"
        m = re.search(r"\.(jpg|jpeg|png|webp)(?:\?|$)", url, re.IGNORECASE)
        if m:
            ext = "." + m.group(1).lower()

        name = hashlib.sha1(url.encode("utf-8", errors="ignore")).hexdigest() + ext
        out_path = temp_dir / name
        if out_path.exists() and out_path.stat().st_size > 0:
            return str(out_path)

        proxy = self._get_proxy()
        base_headers = {
            "User-Agent": USER_AGENT,
            "Referer": (referer or "https://exhentai.org/"),
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        }
        batch = host_fail is not None
        if fast:
            timeout = aiohttp.ClientTimeout(total=30, connect=8, sock_read=20)
            max_attempts = 5
        else:
            timeout = aiohttp.ClientTimeout(total=(60 if batch else 90), connect=(10 if batch else 20), sock_read=(45 if batch else 60))
            max_attempts = 6 if batch else 6

        base_host = urlsplit(url).netloc.lower()
        if host_fail is not None and base_host and host_fail.get(base_host, 0) >= 20:
            return None

        for attempt in range(max_attempts):
            cur = url
            if ".hath.network" in cur and cur.startswith("http://"):
                cur = "https://" + cur[7:]
            parsed = urlsplit(cur)
            has_explicit_port = parsed.port is not None
            if attempt >= 3 and ".hath.network" in cur and not has_explicit_port and cur.startswith("https://"):
                cur = "http://" + cur[8:]

            is_fullimg = "exhentai.org/fullimg/" in cur
            cur_host = (urlsplit(cur).hostname or "").lower()
            is_hath_host = cur_host.endswith(".hath.network")
            req_headers = dict(base_headers)
            if is_fullimg:
                req_headers["Cookie"] = self._get_cookie_header()
            if is_hath_host:
                p_gdl = await self.download_image_via_gallery_dl(
                    cur, out_path, req_headers, proxy=None, fast=fast
                )
                if p_gdl:
                    return p_gdl
                h = urlsplit(cur).netloc.lower()
                if host_fail is not None and h:
                    host_fail[h] = host_fail.get(h, 0) + 1
                if attempt == max_attempts - 1 and host_fail is None:
                    self._logger.warning("下载H@H图片失败: gallery-dl exhausted")
                await asyncio.sleep(0.3 * (attempt + 1))
                continue

            cur_proxy = proxy if force_proxy else (None if is_hath_host else proxy)

            async def _save_from_resp(resp):
                if is_fullimg and resp.status in (301, 302, 303, 307, 308):
                    loc = resp.headers.get("Location") or resp.headers.get("location")
                    if loc and loc.startswith("//"):
                        loc = "https:" + loc
                    if loc and loc.startswith("http"):
                        return await self.download_image(loc, temp_dir=temp_dir, referer=referer, session=session, host_fail=host_fail, force_proxy=force_proxy)
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")
                real_url = str(getattr(resp, "url", "") or "")
                if ex_parser.is_509_marker_url(real_url):
                    raise RuntimeError("HTTP 509 marker")

                part = out_path.with_suffix(out_path.suffix + ".part")
                wrote = 0
                with part.open("wb") as f:
                    async for chunk in resp.content.iter_chunked(128 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        wrote += len(chunk)

                expect = resp.headers.get("Content-Length")
                if expect and expect.isdigit() and wrote < int(expect):
                    try:
                        part.unlink(missing_ok=True)
                    except Exception:
                        pass
                    raise RuntimeError(f"short body {wrote}/{expect}")
                if wrote < 1024:
                    try:
                        part.unlink(missing_ok=True)
                    except Exception:
                        pass
                    raise RuntimeError("empty or too small body")
                part.replace(out_path)
                return str(out_path)

            try:
                if session is not None and not is_hath_host:
                    async with session.get(cur, headers=req_headers, ssl=False, proxy=cur_proxy, allow_redirects=not is_fullimg, timeout=timeout) as resp:
                        return await _save_from_resp(resp)
                async with aiohttp.ClientSession(timeout=timeout, trust_env=True, connector=aiohttp.TCPConnector(ssl=False, family=socket.AF_INET, enable_cleanup_closed=True)) as sess:
                    async with sess.get(cur, headers=req_headers, ssl=False, proxy=cur_proxy, allow_redirects=not is_fullimg) as resp:
                        return await _save_from_resp(resp)
            except Exception as e:
                h = urlsplit(cur).netloc.lower()
                if host_fail is not None and h:
                    host_fail[h] = host_fail.get(h, 0) + 1
                if "509 marker" in str(e):
                    return None
                if attempt == max_attempts - 1 and host_fail is None:
                    self._logger.warning(f"下载图片失败: {e}")
                await asyncio.sleep(0.3 * (attempt + 1))

        return None

    async def resolve_view_candidate(
        self,
        view: list[str],
        nl: str | None = None,
        session: aiohttp.ClientSession | None = None,
        show_key: str | None = None,
        previous_view: list[str] | None = None,
        force_html: bool = False,
    ) -> dict[str, Any]:
        base = ex_parser.view_base(view)
        result: dict[str, Any] = {
            "sample": "",
            "origin": "",
            "nl": "",
            "referer": base,
            "show_key": str(show_key or ""),
            "api_failed": False,
        }
        if not isinstance(view, list) or len(view) < 2:
            return result

        token = str(view[0])
        page_id = str(view[1])
        proxy = self._get_proxy()
        url = base + ("?nl=" + quote(str(nl), safe="") if nl else "")

        if proxy and str(proxy).lower().startswith("socks"):
            data = await self._call_bridge(
                "resolve_views",
                {"views": [[token, page_id, nl] if nl else [token, page_id]]},
            )
            cand = list(data.get("candidates", []) or [])
            if cand:
                c = cand[0] or {}
                result.update(
                    sample=ex_parser.normalize_img_url(c.get("sample") or ""),
                    origin=ex_parser.normalize_img_url(c.get("origin") or ""),
                    nl=str(c.get("nl") or "").strip(),
                    referer=str(c.get("referer") or base).strip() or base,
                )
            return result

        cur_show = str(show_key or "").strip()
        if cur_show and not force_html:
            gid, page_no = ex_parser.split_gid_page(page_id)
            if gid and page_no is not None:
                api_referer = ex_parser.view_base(previous_view) if previous_view else ""
                api_headers = {
                    "User-Agent": USER_AGENT,
                    "Cookie": self._get_cookie_header(),
                    "Accept": "application/json,text/javascript,*/*;q=0.8",
                    "X-Requested-With": "XMLHttpRequest",
                    "Origin": "https://exhentai.org",
                }
                if api_referer:
                    api_headers["Referer"] = api_referer
                api_payload = {
                    "method": "showpage",
                    "gid": int(gid),
                    "page": page_no,
                    "imgkey": token,
                    "showkey": cur_show,
                }
                api_timeout = aiohttp.ClientTimeout(total=10, connect=5, sock_read=8)
                try:
                    if session is not None:
                        async with session.post(
                            "https://exhentai.org/api.php",
                            json=api_payload,
                            headers=api_headers,
                            ssl=False,
                            proxy=proxy,
                            timeout=api_timeout,
                        ) as resp:
                            if resp.status != 200:
                                raise RuntimeError(f"HTTP {resp.status}")
                            raw = await resp.text(errors="ignore")
                    else:
                        async with aiohttp.ClientSession(timeout=api_timeout, headers=api_headers, trust_env=True) as sess:
                            async with sess.post("https://exhentai.org/api.php", json=api_payload, ssl=False, proxy=proxy) as resp:
                                if resp.status != 200:
                                    raise RuntimeError(f"HTTP {resp.status}")
                                raw = await resp.text(errors="ignore")
                    obj = json.loads(raw)
                    err = str(obj.get("error") or "").strip()
                    if err:
                        result["api_failed"] = True
                        raise RuntimeError(err)

                    i3 = str(obj.get("i3") or "")
                    i6 = str(obj.get("i6") or "")
                    i7 = str(obj.get("i7") or "")
                    sample = ex_parser.extract_img_src(i3)
                    origin = ex_parser.extract_origin_url(i6)
                    nl_key = ex_parser.extract_nl_key(i7) or ex_parser.extract_nl_key(i3)
                    if origin and nl_key and "nl=" not in origin:
                        origin += ("&" if "?" in origin else "?") + "nl=" + quote(nl_key, safe="")

                    if sample or origin:
                        result.update(sample=sample, origin=origin, nl=nl_key, referer=base, show_key=cur_show)
                        return result
                except Exception as e:
                    result["api_failed"] = True

        timeout = aiohttp.ClientTimeout(total=15, connect=5, sock_read=12)
        headers = {
            "User-Agent": USER_AGENT,
            "Cookie": self._get_cookie_header(),
            "Referer": (ex_parser.view_base(previous_view) if previous_view else "https://exhentai.org/"),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        if session is not None:
            async with session.get(url, headers=headers, ssl=False, proxy=proxy, timeout=timeout) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")
                text = await resp.text(errors="ignore")
        else:
            async with aiohttp.ClientSession(timeout=timeout, headers=headers, trust_env=True) as sess:
                async with sess.get(url, ssl=False, proxy=proxy) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"HTTP {resp.status}")
                    text = await resp.text(errors="ignore")

        sample = ex_parser.extract_img_src(text)
        nl2 = ex_parser.extract_nl_key(text)
        show2 = ex_parser.extract_show_key(text)
        origin = ex_parser.extract_origin_url(text)
        if origin and nl2 and "nl=" not in origin:
            origin += ("&" if "?" in origin else "?") + "nl=" + quote(nl2, safe="")

        result.update(sample=sample, origin=origin, nl=nl2, referer=base, show_key=(show2 or cur_show))
        return result

    async def download_gallery_images(
        self,
        quality: str,
        href: list[str],
        data: dict[str, Any],
        img_dir: Path,
        session: aiohttp.ClientSession,
        host_fail: dict[str, int] | None = None,
    ) -> tuple[list[tuple[int, str]], list[int], int]:
        q = str(quality or "").lower().strip()
        if q == "mid":
            views = list(data.get("view_hrefs", []) or [])
            if not views:
                return [], [], 0
            return await self.download_mid_views_eh(views, img_dir, session, host_fail)

        raw = list(data.get("thumbnails") or [])
        if not raw:
            info = data.get("info", {}) or {}
            cov = info.get("cover")
            if cov:
                raw.append(cov)

        previews = [str(u) for u in raw if isinstance(u, str) and u]
        candidates = list(data.get("image_candidates", []) or [])

        referers = ["https://exhentai.org/"] * len(previews)
        if candidates:
            previews = [str((it.get("origin") or it.get("best") or it.get("sample") or "")).strip() for it in candidates]
            referers = [str((it.get("referer") or "https://exhentai.org/")).strip() for it in candidates]
        elif q == "low":
            referers = [f"https://exhentai.org/g/{href[0]}/{href[1]}/"] * len(previews)

        if not previews:
            return [], [], 0

        sem = asyncio.Semaphore(2 if q == "high" else 8)

        async def _one(idx: int, url: str):
            async with sem:
                ref = referers[idx - 1] if idx - 1 < len(referers) else "https://exhentai.org/"
                p1 = await self.download_image(url, temp_dir=img_dir, referer=ref, session=session, host_fail=host_fail, fast=(q != "high"))
                if p1:
                    return p1

                if candidates and idx - 1 < len(candidates):
                    s2 = str((candidates[idx - 1].get("sample") or "")).strip()
                    if s2 and s2 != url:
                        p2 = await self.download_image(s2, temp_dir=img_dir, referer=ref, session=session, host_fail=host_fail, fast=(q != "high"))
                        if p2:
                            return p2
                return None


        paths = await asyncio.gather(*[_one(i + 1, u) for i, u in enumerate(previews)], return_exceptions=True)
        ok: list[tuple[int, str]] = []
        fail_idx: list[int] = []
        for idx, pp in enumerate(paths, 1):
            if isinstance(pp, str) and pp:
                ok.append((idx, pp))
            else:
                fail_idx.append(idx)
        return ok, fail_idx, len(previews)

    async def download_mid_views_eh(
        self,
        views: list[list[str]],
        img_dir: Path,
        session: aiohttp.ClientSession,
        host_fail: dict[str, int] | None = None,
    ) -> tuple[list[tuple[int, str]], list[int], int]:
        total = len(views)
        if total <= 0:
            return [], [], 0

        max_retry = 3
        max_skip_hath_keys = 12
        sem = asyncio.Semaphore(3)
        batch_t0 = time.perf_counter()

        self._logger.info(
            f"[exapi_cosmos][mid-prof] start total={total} workers=3 max_retry={max_retry} max_skip_keys={max_skip_hath_keys}"
        )

        shared_show: dict[str, str] = {"value": ""}
        show_lock = asyncio.Lock()
        stat_lock = asyncio.Lock()
        stats: dict[str, Any] = {
            "attempts": 0,
            "done": 0,
            "page_ok": 0,
            "page_fail": 0,
            "resolve_calls": 0,
            "resolve_ex": 0,
            "resolve_ms": 0.0,
            "download_calls": 0,
            "download_ok": 0,
            "download_fail": 0,
            "download_ms": 0.0,
            "api_failed": 0,
            "switch_nl": 0,
            "leak_breaks": 0,
            "marker509": 0,
        }
        slow_pages: list[tuple[int, float, int, int]] = []

        first_prefetched: dict[str, Any] | None = None
        try:
            first_prefetched = await self.resolve_view_candidate(views[0], session=session, force_html=True)
            sk = str((first_prefetched or {}).get("show_key") or "").strip()
            if sk:
                shared_show["value"] = sk
        except Exception as e:
            self._logger.warning(f"mid预热第一页失败: {e}")

        async def _one(idx: int, view: list[str], retry_limit: int = max_retry, force_html_first: bool = False, skip_hath_limit: int = max_skip_hath_keys, local_host_fail: dict[str, int] | None = None, count_done: bool = True) -> tuple[int, str | None]:
            async with sem:
                page_t0 = time.perf_counter()
                used_attempt = 0
                cur_nl: str | None = None
                used_nl: set[str] = set()
                force_html = force_html_first
                leak_skip_hath_key = False
                local_show = str(shared_show.get("value") or "").strip()
                prefetched = first_prefetched if idx == 0 else None

                async def _finish(path: str | None) -> tuple[int, str | None]:
                    if not count_done:
                        return idx + 1, path
                    page_ms = (time.perf_counter() - page_t0) * 1000.0
                    async with stat_lock:
                        stats["done"] += 1
                        if path:
                            stats["page_ok"] += 1
                        else:
                            stats["page_fail"] += 1
                        done = int(stats["done"])
                        ok_done = int(stats["page_ok"])
                        fail_done = int(stats["page_fail"])
                        slow_pages.append((idx + 1, round(page_ms, 1), used_attempt, 1 if path else 0))
                    if done % 10 == 0 or done == total:
                        self._logger.info(
                            f"[exapi_cosmos][mid-progress] done={done}/{total} page_ok={ok_done} page_fail={fail_done}"
                        )
                    return idx + 1, path

                for attempt in range(retry_limit):
                    used_attempt = attempt + 1
                    async with stat_lock:
                        stats["attempts"] += 1


                    try:
                        resolve_t0 = time.perf_counter()
                        if prefetched is not None:
                            cand = prefetched
                            prefetched = None
                        else:
                            if not local_show:
                                local_show = str(shared_show.get("value") or "").strip()
                            prev_view = views[idx - 1] if idx > 0 else None
                            cand = await self.resolve_view_candidate(
                                view,
                                nl=cur_nl,
                                session=session,
                                show_key=(local_show or None),
                                previous_view=prev_view,
                                force_html=(force_html or bool(cur_nl)),
                            )
                        resolve_ms = (time.perf_counter() - resolve_t0) * 1000.0
                        async with stat_lock:
                            stats["resolve_calls"] += 1
                            stats["resolve_ms"] += resolve_ms
                    except Exception:
                        async with stat_lock:
                            stats["resolve_ex"] += 1
                        await asyncio.sleep(0.2 * (attempt + 1))
                        continue

                    got_show = str(cand.get("show_key") or "").strip()
                    if got_show:
                        local_show = got_show
                        async with show_lock:
                            if shared_show.get("value") != got_show:
                                shared_show["value"] = got_show

                    if bool(cand.get("api_failed")):
                        async with stat_lock:
                            stats["api_failed"] += 1
                        force_html = True
                        local_show = ""
                        async with show_lock:
                            shared_show["value"] = ""

                    sample_url = str(cand.get("sample") or "").strip()
                    ref = str(cand.get("referer") or ex_parser.view_base(view) or "https://exhentai.org/").strip() or "https://exhentai.org/"

                    if sample_url and ex_parser.is_509_marker_url(sample_url):
                        async with stat_lock:
                            stats["marker509"] += 1
                        self._logger.warning(f"[exapi_cosmos] 检测到509占位图，终止该页重试 idx={idx + 1}")
                        break

                    if sample_url:
                        dl_t0 = time.perf_counter()
                        p = await self.download_image(
                            sample_url,
                            temp_dir=img_dir,
                            referer=ref,
                            session=session,
                            host_fail=(local_host_fail if local_host_fail is not None else host_fail),
                            fast=False,
                        )
                        dl_ms = (time.perf_counter() - dl_t0) * 1000.0
                        async with stat_lock:
                            stats["download_calls"] += 1
                            stats["download_ms"] += dl_ms
                            if p:
                                stats["download_ok"] += 1
                            else:
                                stats["download_fail"] += 1
                        if not p:
                            p = await self.download_image(sample_url, temp_dir=img_dir, referer=ref, session=None, host_fail=(local_host_fail if local_host_fail is not None else host_fail), fast=False)
                        if p:
                            return await _finish(p)


                    next_nl = str(cand.get("nl") or "").strip()
                    if next_nl and next_nl not in used_nl and len(used_nl) < skip_hath_limit:
                        used_nl.add(next_nl)
                        cur_nl = next_nl
                        force_html = True
                        async with stat_lock:
                            stats["switch_nl"] += 1
                    else:
                        if (not next_nl) or (next_nl in used_nl) or (len(used_nl) >= skip_hath_limit):
                            leak_skip_hath_key = True
                        cur_nl = None
                        if local_show and not force_html:
                            force_html = True
                        elif force_html and leak_skip_hath_key:
                            async with stat_lock:
                                stats["leak_breaks"] += 1
                            break

                    await asyncio.sleep(0.15 * (attempt + 1))

                return await _finish(None)

        rs = await asyncio.gather(*[_one(i, v) for i, v in enumerate(views)], return_exceptions=True)
        ok: list[tuple[int, str]] = []
        fail_idx: list[int] = []
        for i, r in enumerate(rs, 1):
            if isinstance(r, Exception):
                fail_idx.append(i)
                continue
            _, p = r
            if isinstance(p, str) and p:
                ok.append((i, p))
            else:
                fail_idx.append(i)
        ok.sort(key=lambda x: x[0])

        if fail_idx:
            self._logger.info(f"[exapi_cosmos][mid-prof] rescue start fail={len(fail_idx)}")
            rescue_max_retry = 2
            rescue_rs = await asyncio.gather(
                *[
                    _one(
                        i - 1,
                        views[i - 1],
                        retry_limit=rescue_max_retry,
                        force_html_first=True,
                        skip_hath_limit=max_skip_hath_keys * 3,
                        local_host_fail={},
                        count_done=False,
                    )
                    for i in fail_idx
                ],
                return_exceptions=True,
            )
            rescue_ok: list[tuple[int, str]] = []
            for r in rescue_rs:
                if isinstance(r, Exception):
                    continue
                page_no, p = r
                if isinstance(p, str) and p:
                    rescue_ok.append((page_no, p))
            if rescue_ok:
                ok_map = {i: p for i, p in ok}
                for i, p in rescue_ok:
                    ok_map[i] = p
                ok = sorted(ok_map.items(), key=lambda x: x[0])
                rescue_set = {i for i, _ in rescue_ok}
                fail_idx = [i for i in fail_idx if i not in rescue_set]
            self._logger.info(f"[exapi_cosmos][mid-prof] rescue done ok={len(ok)} fail={len(fail_idx)}")

        elapsed_s = round(time.perf_counter() - batch_t0, 3)
        resolve_calls = int(stats["resolve_calls"]) or 1
        download_calls = int(stats["download_calls"]) or 1
        avg_resolve_ms = round(float(stats["resolve_ms"]) / resolve_calls, 2)
        avg_download_ms = round(float(stats["download_ms"]) / download_calls, 2)
        top_slow = sorted(slow_pages, key=lambda x: x[1], reverse=True)[:5]
        self._logger.info(
            "[exapi_cosmos][mid-prof] "
            f"done total={total} ok={len(ok)} fail={len(fail_idx)} elapsed={elapsed_s}s "
            f"attempts={int(stats['attempts'])} resolve_calls={int(stats['resolve_calls'])} resolve_ex={int(stats['resolve_ex'])} "
            f"avg_resolve_ms={avg_resolve_ms} download_calls={int(stats['download_calls'])} "
            f"download_ok={int(stats['download_ok'])} download_fail={int(stats['download_fail'])} avg_download_ms={avg_download_ms} "
            f"api_failed={int(stats['api_failed'])} switch_nl={int(stats['switch_nl'])} leak_breaks={int(stats['leak_breaks'])} marker509={int(stats['marker509'])} "
            f"top_slow={top_slow}"
        )

        return ok, fail_idx, total
