from __future__ import annotations

import asyncio
import aiohttp
import hashlib
import html
import json
import os
import re
import shlex
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.parse import quote, urlsplit
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
from astrbot.api.message_components import File, Image, Node, Nodes, Plain

@register("exapi_cosmos", "zhanzhao2", "exHentai 搜索插件（基于 exApi）", "0.2.1", "https://github.com/zhanzhao2/astrbot_plugin_exapi_cosmos")
class ExApiCosmosPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig = None):
        super().__init__(context)
        self.config = config or {}
        self.bridge = Path(__file__).parent / "node" / "bridge.js"
        self._last_items: dict[str, list[dict[str, Any]]] = {}
        self._zip_pending: dict[str, dict[str, Any]] = {}
        try:
            self._cleanup_orphans()
            asyncio.get_running_loop().create_task(self._cleanup_job())
        except RuntimeError:
            pass
        except Exception:
            pass
        logger.info("exApi-Cosmos 插件初始化完成")

    def _cookies(self) -> dict[str, str]:
        return {
            "ipb_member_id": str(self.config.get("ipb_member_id", "")).strip(),
            "ipb_pass_hash": str(self.config.get("ipb_pass_hash", "")).strip(),
            "igneous": str(self.config.get("igneous", "")).strip(),
        }

    def _cookie_header(self) -> str:
        c = self._cookies()
        return f"ipb_member_id={c['ipb_member_id']}; ipb_pass_hash={c['ipb_pass_hash']}; igneous={c['igneous']}"

    def _missing(self) -> list[str]:
        c = self._cookies()
        return [k for k, v in c.items() if not v]

    def _proxy(self) -> str | None:
        if not bool(self.config.get("use_proxy", False)):
            return None
        p = str(self.config.get("proxy_url", "")).strip()
        return p or None

    def _page_size(self) -> int:
        n = int(self.config.get("search_page_size", 5) or 5)
        return max(1, min(n, 20))

    def _timeout(self) -> int:
        n = int(self.config.get("request_timeout", 90) or 90)
        return max(10, min(n, 300))

    def _cache_key(self, event: AstrMessageEvent) -> str:
        return f"{event.get_platform_id()}:{event.get_session_id()}"

    def _temp_root(self) -> Path:
        for root in (Path("/shared_files/exapi_cosmos"), Path(__file__).parent / "_tmp"):
            try:
                root.mkdir(parents=True, exist_ok=True)
                return root
            except Exception:
                pass
        return Path(__file__).parent

    def _cleanup_orphans(self, max_age_sec: int = 6 * 3600):
        try:
            root = self._temp_root()
            now = time.time()
            for p in root.iterdir():
                if p.is_dir() and p.name.startswith(("exzip_", "exprev_", "eximg_")) and now - p.stat().st_mtime > max_age_sec:
                    shutil.rmtree(p, ignore_errors=True)
            tmp = root / "temp_img"
            if tmp.is_dir():
                for f in tmp.iterdir():
                    if f.is_file() and now - f.stat().st_mtime > max_age_sec:
                        try: f.unlink()
                        except Exception: pass
        except Exception:
            pass

    async def _cleanup_job(self):
        while True:
            await asyncio.sleep(3600)
            self._cleanup_orphans()

    def _zip_limit(self) -> int:
        n = int(self.config.get("zip_image_limit", 500) or 500)
        return max(10, min(n, 1200))

    def _set_zip_pending(self, event: AstrMessageEvent, href: list[str]):
        self._zip_pending[self._cache_key(event)] = {"href": [str(href[0]), str(href[1])]}

    def _get_zip_pending(self, event: AstrMessageEvent) -> dict[str, Any] | None:
        return self._zip_pending.get(self._cache_key(event))

    def _clear_zip_pending(self, event: AstrMessageEvent):
        self._zip_pending.pop(self._cache_key(event), None)

    def _save_last_items(self, event: AstrMessageEvent, items: list[dict[str, Any]]):
        rows: list[dict[str, Any]] = []
        for it in items[: self._page_size()]:
            h = it.get("href") or []
            if isinstance(h, list) and len(h) >= 2:
                rows.append({"href": [str(h[0]), str(h[1])], "title": str(it.get("title", "未知标题"))})
        self._last_items[self._cache_key(event)] = rows

    def _pick_cached_href(self, event: AstrMessageEvent, idx: int) -> list[str] | None:
        arr = self._last_items.get(self._cache_key(event), [])
        if 1 <= idx <= len(arr):
            h = arr[idx - 1].get("href") or []
            if isinstance(h, list) and len(h) >= 2:
                return [str(h[0]), str(h[1])]
        return None

    def _preview_limit(self) -> int:
        n = int(self.config.get("preview_limit", 200) or 200)
        return max(1, min(n, 200))

    async def _cleanup_files_later(self, paths: list[str], delay: int = 90):
        await asyncio.sleep(delay)
        for path in paths:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

    async def _cleanup_dir_later(self, path: str, delay: int = 120):
        await asyncio.sleep(delay)
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass



    async def _download_image(
        self,
        url: str,
        temp_dir: Path | None = None,
        referer: str | None = None,
        session: aiohttp.ClientSession | None = None,
        host_fail: dict[str, int] | None = None,
    ) -> str | None:
        """下载图片到本地临时目录，返回本地路径；失败返回 None。"""
        if not isinstance(url, str) or not url.strip():
            return None

        url = url.strip()
        if url.startswith("//"):
            url = "https:" + url
        if not url.startswith("http"):
            return None
        if url.startswith("http://") and ".hath.network" in url:
            url = "https://" + url[7:]

        if temp_dir is None:
            temp_dir = self._temp_root() / "temp_img"
        temp_dir.mkdir(parents=True, exist_ok=True)

        ext = ".jpg"
        m = re.search(r"\.(jpg|jpeg|png|webp)(?:\?|$)", url, re.IGNORECASE)
        if m:
            ext = "." + m.group(1).lower()

        name = hashlib.sha1(url.encode("utf-8", errors="ignore")).hexdigest() + ext
        out_path = temp_dir / name
        if out_path.exists() and out_path.stat().st_size > 0:
            return str(out_path)

        proxy = self._proxy()
        base_headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": (referer or "https://exhentai.org/"),
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        }
        batch = host_fail is not None
        timeout = aiohttp.ClientTimeout(total=(60 if batch else 90), connect=(10 if batch else 20), sock_read=(45 if batch else 60))
        base_host = urlsplit(url).netloc.lower()
        if host_fail is not None and base_host and host_fail.get(base_host, 0) >= 6:
            return None

        max_attempts = 3 if batch else 6
        for attempt in range(max_attempts):
            cur = url
            if attempt >= 3 and ".hath.network" in cur:
                if cur.startswith("https://"):
                    cur = "http://" + cur[8:]
                elif cur.startswith("http://"):
                    cur = "https://" + cur[7:]

            is_fullimg = "exhentai.org/fullimg/" in cur
            req_headers = dict(base_headers)
            if is_fullimg:
                req_headers["Cookie"] = self._cookie_header()

            async def _save_from_resp(resp):
                if is_fullimg and resp.status in (301, 302, 303, 307, 308):
                    loc = resp.headers.get("Location") or resp.headers.get("location")
                    if loc and loc.startswith("//"):
                        loc = "https:" + loc
                    if loc and loc.startswith("http"):
                        return await self._download_image(loc, temp_dir=temp_dir, referer=referer, session=session, host_fail=host_fail)
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")

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
                if session is not None:
                    async with session.get(cur, headers=req_headers, ssl=False, proxy=proxy, allow_redirects=not is_fullimg, timeout=timeout) as resp:
                        return await _save_from_resp(resp)
                async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as sess:
                    async with sess.get(cur, headers=req_headers, ssl=False, proxy=proxy, allow_redirects=not is_fullimg) as resp:
                        return await _save_from_resp(resp)
            except Exception as e:
                h = urlsplit(cur).netloc.lower()
                if host_fail is not None and h:
                    host_fail[h] = host_fail.get(h, 0) + 1
                if attempt == 5 and host_fail is None:
                    logger.warning(f"下载图片失败: {e}")
                await asyncio.sleep(0.3 * (attempt + 1))

        return None


    async def _resolve_view_candidate(self, view: list[str], nl: str | None = None, session: aiohttp.ClientSession | None = None) -> dict[str, str]:
        if not isinstance(view, list) or len(view) < 2:
            return {"sample": "", "nl": "", "referer": "https://exhentai.org/"}

        token = str(view[0])
        page_id = str(view[1])
        base = f"https://exhentai.org/s/{token}/{page_id}"
        url = base
        if nl:
            url = base + "?nl=" + quote(str(nl), safe="")
        proxy = self._proxy()
        if proxy and proxy.lower().startswith("socks"):
            data = await self._call(
                "resolve_views",
                {"views": [[token, page_id, nl] if nl else [token, page_id]]},
            )
            cand = list(data.get("candidates", []) or [])
            if cand:
                c = cand[0] or {}
                return {
                    "sample": str((c.get("sample") or c.get("best") or "")).strip(),
                    "nl": str((c.get("nl") or "")).strip(),
                    "referer": str((c.get("referer") or base)).strip() or base,
                }
            return {"sample": "", "nl": "", "referer": base}
        timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=15)
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Cookie": self._cookie_header(),
            "Referer": "https://exhentai.org/",
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

        m = re.search(r'<img[^>]*id="img"[^>]*src="([^"]+)"', text, re.IGNORECASE)
        if not m:
            m = re.search(r'<img[^>]*src="([^"]+)"[^>]*id="img"', text, re.IGNORECASE)
        if not m:
            m = re.search(r'<img[^>]*src="([^"]+)"[^>]*style', text, re.IGNORECASE)
        sample = html.unescape(m.group(1).strip()) if m else ""

        if sample.startswith("//"):
            sample = "https:" + sample
        elif sample.startswith("/"):
            sample = "https://exhentai.org" + sample

        m2 = re.search(r'onclick="return nl\(\'([^\)]+)\'\)', text, re.IGNORECASE)
        if not m2:
            m2 = re.search(r"nl\('([^\)]+)'\)", text, re.IGNORECASE)
        nl2 = html.unescape(m2.group(1).strip()) if m2 else ""
        return {"sample": sample, "nl": nl2, "referer": base}

    async def _send_previews(self, event: AstrMessageEvent, previews: list[str]):
        """并发下载后，最终只发送一条合并消息。"""
        if not previews:
            return

        uin = str(event.get_self_id() or event.get_sender_id() or "0")
        tmp_dir = Path(tempfile.mkdtemp(prefix="exprev_", dir=str(self._temp_root())))

        sem = asyncio.Semaphore(2)

        async def _one(idx: int, url: str):
            async with sem:
                p = await self._download_image(url, temp_dir=tmp_dir)
                return idx, p

        tasks = [_one(i + 1, u) for i, u in enumerate(previews)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        ok: list[tuple[int, str]] = []
        fail = 0
        for r in results:
            if isinstance(r, Exception):
                fail += 1
                continue
            idx, p = r
            if p:
                ok.append((idx, p))
            else:
                fail += 1
        ok.sort(key=lambda x: x[0])
        used_paths = [p for _, p in ok]

        if not ok:
            yield event.plain_result(f"⚠️ 预览图全部下载失败（共 {len(previews)} 张）")
            asyncio.create_task(self._cleanup_dir_later(str(tmp_dir), 10))
            return

        nodes = [Node(name="exApi-Cosmos", uin=uin, content=[Plain(f"🖼️ 预览图 {len(ok)}/{len(previews)} 张（失败 {fail}）")])]
        for idx, p in ok:
            nodes.append(Node(name="exApi-Cosmos", uin=uin, content=[Plain(f"#{idx}"), Image.fromFileSystem(p)]))

        try:
            yield event.chain_result([Nodes(nodes)])
        except Exception as e:
            logger.warning(f"Nodes 发送失败，回退图片链: {e}")
            chain = [Plain(f"🖼️ 预览图 {len(ok)}/{len(previews)} 张（失败 {fail}）")]
            for idx, p in ok:
                chain.append(Plain(f"\n#{idx}"))
                chain.append(Image.fromFileSystem(p))
            yield event.chain_result(chain)

        asyncio.create_task(self._cleanup_files_later(used_paths, 90))
        asyncio.create_task(self._cleanup_dir_later(str(tmp_dir), 120))

    async def _send_images_nodes_batched(
        self,
        event: AstrMessageEvent,
        items: list[tuple[int, str]],
        total: int,
        fail: int,
        batch_size: int = 20,
    ):
        if not items:
            return
        uin = str(event.get_self_id() or event.get_sender_id() or "0")
        items = sorted(items, key=lambda x: x[0])

        for pos in range(0, len(items), batch_size):
            part = items[pos: pos + batch_size]
            first_idx = part[0][0]
            last_idx = part[-1][0]
            nodes = [Node(name="exApi-Cosmos", uin=uin, content=[Plain(f"🖼️ 图片 {first_idx}-{last_idx}/{total}（失败 {fail}）")])]
            for idx, p in part:
                nodes.append(Node(name="exApi-Cosmos", uin=uin, content=[Plain(f"#{idx}"), Image.fromFileSystem(p)]))

            try:
                yield event.chain_result([Nodes(nodes)])
            except Exception as e:
                logger.warning(f"图片合并消息发送失败，回退图片链: {e}")
                chain = [Plain(f"🖼️ 图片 {first_idx}-{last_idx}/{total}（失败 {fail}）")]
                for idx, p in part:
                    chain.append(Plain(f"\n#{idx}"))
                    chain.append(Image.fromFileSystem(p))
                yield event.chain_result(chain)

            if pos + batch_size < len(items):
                await asyncio.sleep(0.8)

    def _args(self, event: AstrMessageEvent) -> list[str]:
        # 兼容 /exs 与 exs 两种输入
        text = (event.message_str or "").strip()
        if not text:
            return []
        try:
            parts = shlex.split(text)
        except ValueError:
            parts = text.split()
        if not parts:
            return []
        known = {"exhelp","exstatus","exhome","exs","ex","exa","exi","exzip","eximg"}
        if parts[0].startswith("@") and len(parts) > 1:
            parts = parts[1:]
        head = parts[0].lstrip("/")
        low = head.lower()
        if low in known:
            return parts[1:]
        for cmd in ["exstatus","exhelp","exhome","exa","exi","exzip","eximg","exs","ex"]:
            if low.startswith(cmd) and low != cmd:
                return [head[len(cmd):]] + parts[1:]
        return parts

    def _parse_href(self, args: list[str]) -> list[str] | None:
        if not args:
            return None
        m = re.search(r"/g/(\d+)/([0-9a-zA-Z]+)/?", " ".join(args))
        if m:
            return [m.group(1), m.group(2)]
        if len(args) >= 2 and args[0].isdigit():
            return [args[0], args[1].strip("/ ")]
        if len(args) == 1:
            s = args[0].strip()
            for sep in ["/", ":", "|", ","]:
                if sep in s:
                    a, b = s.split(sep, 1)
                    if a.strip().isdigit() and b.strip():
                        return [a.strip(), b.strip("/ ")]
        return None

    def _parse_advanced(self, args: list[str]) -> tuple[dict[str, Any], int]:
        cfg: dict[str, Any] = {"advanced": {"enable": {"name": True, "tags": True}}}
        page = 1
        ns = {"language","artist","group","parody","character","female","male","mixed","other","cosplayer","uploader"}
        for token in args:
            if "=" not in token:
                continue
            k, v = token.split("=", 1)
            k = k.strip().lower()
            v = v.strip()
            if not v:
                continue
            if k in {"q", "text", "keyword"}:
                cfg["text"] = v
            elif k in {"page", "p"}:
                page = max(1, int(v) if v.isdigit() else 1)
            elif k in {"type", "types"}:
                cfg["type"] = [x.strip() for x in v.split(",") if x.strip()]
            elif k.startswith("tag.") or k in ns:
                n = k[4:] if k.startswith("tag.") else k
                cfg.setdefault("tag", {})[n] = [x.strip() for x in v.split(",") if x.strip()]
            elif k in {"rating", "rate"} and v.isdigit():
                r = int(v)
                if 2 <= r <= 5:
                    cfg.setdefault("advanced", {})["rating"] = r
            elif k in {"between", "pages"}:
                p = re.split(r"[-,:]", v, maxsplit=1)
                if len(p) == 2 and p[0].isdigit() and p[1].isdigit():
                    cfg.setdefault("advanced", {})["between"] = [int(p[0]), int(p[1])]
        if "tag" in cfg and not cfg["tag"]:
            cfg.pop("tag", None)
        return cfg, page

    async def _call(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.bridge.exists():
            raise RuntimeError("bridge.js 不存在")
        m = self._missing()
        if m:
            raise RuntimeError("缺少 Cookie 字段: " + ", ".join(m))
        body = {"action": action, "cookies": self._cookies(), "proxy": self._proxy(), **payload}
        p = await asyncio.create_subprocess_exec(
            "node", str(self.bridge),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.bridge.parent),
        )
        try:
            out, err = await asyncio.wait_for(p.communicate(json.dumps(body).encode("utf-8")), timeout=self._timeout())
        except asyncio.TimeoutError:
            p.kill()
            raise RuntimeError("请求超时")
        text = out.decode("utf-8", errors="ignore").strip()
        if not text:
            raise RuntimeError("bridge 无返回")
        data = json.loads(text.splitlines()[-1])
        if not data.get("ok", False):
            raise RuntimeError(data.get("error", "未知错误"))
        return data.get("data", {})

    def _fmt_list(self, title: str, items: list[dict[str, Any]], page: int, pages: int, nxt: str) -> str:
        if not items:
            return title + "\n\n📭 当前页没有结果"
        lines = [title, "━━━━━━━━━━━━━━━━━━━━━"]
        for i, it in enumerate(items[: self._page_size()], 1):
            h = it.get("href") or []
            gid = f"{h[0]}/{h[1]}" if isinstance(h, list) and len(h) >= 2 else "N/A"
            t = str(it.get("title", "未知标题"))[:52]
            lines.append(f"{i}. 【{gid}】{t}")
        lines += ["", "━━━━━━━━━━━━━━━━━━━━━", f"📄 第 {page} 页", f"💡 下一页: {nxt}", "💡 详情: /exi <序号|gid/token>"]
        if pages and pages > 0:
            lines.insert(-3, f"📚 总页: {pages}")
        return "\n".join(lines)

    def _fmt_info(self, href: list[str], info: dict[str, Any], pages: int) -> str:
        title = info.get("title", "未知标题")
        if isinstance(title, list):
            title = title[0] if title else "未知标题"
        lines = [f"📖 {title}", "━━━━━━━━━━━━━━━━━━━━━", f"🆔 {href[0]}/{href[1]}"]
        for k, n in [("type", "类型"), ("uploader", "上传者"), ("published", "发布时间"), ("language", "语言"), ("size", "大小"), ("length", "页数")]: 
            if info.get(k): lines.append(f"{n}: {info.get(k)}")
        if pages and pages > 1: lines.append(f"缩略图分页: {pages}")
        lines.append("━━━━━━━━━━━━━━━━━━━━━")
        return "\n".join(lines)

    @filter.regex(r"^/?exhelp$")
    async def exhelp(self, event: AstrMessageEvent):
        yield event.plain_result(
            "📚 exApi-Cosmos 命令\n\n"
            "/exstatus - 检查状态\n"
            "/exhome [页码] - 首页\n"
            "/exs 或 /ex <关键词> [页码] - 搜索\n"
            "/exa key=value ... - 高级搜索（/exa help）\n"
            "/exi <序号|gid/token|URL> - 详情\n"
            "/exzip high|mid|low|no - 发送完整压缩包\n"
            "/eximg high|mid|low|no - 分批合并消息发图（每条最多20张）\n"
            "/exhelp - 帮助"
        )

    @filter.regex(r"^/?exstatus$")
    async def exstatus(self, event: AstrMessageEvent):
        logger.info(f"[exapi_cosmos] exstatus called: {event.message_str}")
        m = self._missing()
        txt = "✅ Cookie 已配置" if not m else ("❌ 缺少 Cookie: " + ", ".join(m))
        proxy = self._proxy()
        txt += "\n🌐 代理: " + (proxy if proxy else "未启用")
        txt += f"\n📄 每页显示: {self._page_size()}"
        txt += f"\n⏱️ 超时: {self._timeout()} 秒"
        yield event.plain_result(txt)

    @filter.regex(r"^/?exhome(?:\s*\d+)?$")
    async def exhome(self, event: AstrMessageEvent):
        logger.info(f"[exapi_cosmos] exhome called: {event.message_str}")
        args = self._args(event)
        page = 1
        if args and args[0].isdigit(): page = max(1, int(args[0]))
        try:
            yield event.plain_result(f"🏠 加载首页第 {page} 页...")
            data = await self._call("index", {"page": page - 1})
            items = data.get("items", [])
            self._save_last_items(event, items)
            pages = int(data.get("pages", 0) or 0)
            now = int(data.get("page", page) or page)
            msg = self._fmt_list(f"🏠 首页 (第{now}页)", items, now, pages, f"/exhome {now+1}")
            yield event.plain_result(msg)
        except Exception as e:
            logger.error(f"exhome失败: {e}")
            yield event.plain_result(f"❌ 首页获取失败: {e}")

    @filter.regex(r"^.*?/?(?:exs|ex)\b.*$")
    async def exs(self, event: AstrMessageEvent):
        logger.info(f"[exapi_cosmos] exs called: {event.message_str}")
        args = self._args(event)
        if not args:
            yield event.plain_result("❌ 用法: /exs <关键词> [页码]")
            return
        page = 1
        if len(args) > 1 and args[-1].isdigit():
            page = max(1, int(args[-1]))
            kw = " ".join(args[:-1])
        else:
            kw = " ".join(args)
        kw = kw.strip()
        if not kw:
            yield event.plain_result("❌ 关键词不能为空")
            return
        try:
            yield event.plain_result(f"🔍 搜索 {kw} 第 {page} 页...")
            data = await self._call("search", {"keyword": kw, "page": page})
            items = data.get("items", [])
            self._save_last_items(event, items)
            pages = int(data.get("pages", 0) or 0)
            now = int(data.get("page", page) or page)
            msg = self._fmt_list(f"🔍 搜索: {kw} (第{now}页)", items, now, pages, f"/exs {kw} {now+1}")
            yield event.plain_result(msg)
        except Exception as e:
            logger.error(f"exs失败: {e}")
            yield event.plain_result(f"❌ 搜索失败: {e}")

    @filter.regex(r"^/?exa(?:\s+.*)?$")
    async def exa(self, event: AstrMessageEvent):
        logger.info(f"[exapi_cosmos] exa called: {event.message_str}")
        args = self._args(event)
        if not args or (len(args) == 1 and args[0].lower() in {"help", "-h", "--help"}):
            yield event.plain_result(
                "🔎 高级搜索: /exa key=value ...\n"
                "示例: /exa text=genshin type=Doujinshi language=chinese rating=4 page=1\n"
                "支持: text/q, type, tag.xxx, language, artist, female, male, rating, between, page"
            )
            return
        if args[0].startswith("{"):
            try:
                cfg = json.loads(" ".join(args))
                page = max(1, int(cfg.pop("page", 1))) if isinstance(cfg, dict) else 1
            except Exception as e:
                yield event.plain_result(f"❌ JSON 解析失败: {e}")
                return
        else:
            cfg, page = self._parse_advanced(args)
        try:
            yield event.plain_result(f"🧠 高级搜索第 {page} 页...")
            data = await self._call("advanced_search", {"config": cfg, "page": page})
            items = data.get("items", [])
            pages = int(data.get("pages", 0) or 0)
            self._save_last_items(event, items)
            now = int(data.get("page", page) or page)
            msg = self._fmt_list(f"🧠 高级搜索 (第{now}页)", items, now, pages, f"/exa page={now+1} ...")
            yield event.plain_result(msg + "\n\n配置: " + json.dumps(cfg, ensure_ascii=False))
        except Exception as e:
            logger.error(f"exa失败: {e}")
            yield event.plain_result(f"❌ 高级搜索失败: {e}")


    @filter.regex(r"^/?exi(?:\s+.+)?$")
    async def exi(self, event: AstrMessageEvent):
        logger.info(f"[exapi_cosmos] exi called: {event.message_str}")
        args = self._args(event)
        if not args:
            yield event.plain_result("❌ 用法: /exi <序号|gid/token|URL>")
            return
        href = self._parse_href(args)
        if not href and len(args) == 1 and args[0].isdigit():
            href = self._pick_cached_href(event, int(args[0]))
        if not href:
            yield event.plain_result("❌ 无法解析详情目标。请先 /exs 搜索后再用 /exi 1，或直接 /exi gid/token")
            return
        try:
            yield event.plain_result(f"📖 获取详情: {href[0]}/{href[1]} ...")
            data = await self._call("gallery", {"href": href, "fetch_all_previews": True, "max_previews": self._preview_limit(), "thumb_size": 0})
            info = data.get("info", {}) or {}
            pages = int(data.get("pages", 0) or 0)

            raw = list(data.get("thumbnails", []) or [])
            if not raw:
                raw = list(data.get("thumbnails", []) or [])
                cov = info.get("cover")
                if cov:
                    raw.append(cov)
                raw.extend(list(data.get("thumbnails", []) or []))

            previews: list[str] = []
            seen: set[str] = set()
            for u in raw:
                if isinstance(u, str) and u and u not in seen:
                    seen.add(u)
                    previews.append(u)

            yield event.plain_result(self._fmt_info(href, info, pages) + f"\n🖼️ 预览图: {len(previews)} 张")

            if previews:
                logger.info(f"[exapi_cosmos] exi preview count={len(previews)} private={event.is_private_chat()}")
                async for r in self._send_previews(event, previews):
                    yield r
            self._set_zip_pending(event, href)
            yield event.plain_result("📦 请选择发送方式与画质：\n压缩包: /exzip high|mid|low\n合并图: /eximg high|mid|low（每条最多20张，分批发送）\n取消: /exzip no")
        except Exception as e:
            logger.error(f"exi失败: {e}")
            yield event.plain_result(f"❌ 获取详情失败: {e}")

    @filter.regex(r"^/?exzip(?:\s+.*)?$")
    async def exzip(self, event: AstrMessageEvent):
        args = self._args(event)
        pending = self._get_zip_pending(event)
        if not pending:
            yield event.plain_result("❌ 当前没有待打包任务。请先用 /exi 查看详情。")
            return
        if not args:
            yield event.plain_result("📦 请选择发送方式与画质：\n压缩包: /exzip high|mid|low\n合并图: /eximg high|mid|low（每条最多20张，分批发送）\n取消: /exzip no")
            return

        op = args[0].lower().strip()
        if op in {"no", "n", "cancel", "取消", "否"}:
            self._clear_zip_pending(event)
            yield event.plain_result("🛑 已取消发送压缩包")
            return

        quality = None
        if op in {"yes", "y", "ok", "确认", "是", "high", "h", "原图"}:
            quality = "high"
        elif op in {"mid", "m", "medium", "resample", "sample", "中"}:
            quality = "mid"
        elif op in {"low", "l", "thumb", "缩略", "低"}:
            quality = "low"

        if not quality:
            yield event.plain_result("❌ 用法: /exzip high|mid|low|no")
            return

        href = pending.get("href")
        self._clear_zip_pending(event)
        if not isinstance(href, list) or len(href) < 2:
            yield event.plain_result("❌ 待打包数据无效，请重新 /exi")
            return

        task_dir = tempfile.mkdtemp(prefix="exzip_", dir=str(self._temp_root()))
        img_dir = Path(task_dir) / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        zip_name = f"ex_{href[0]}_{href[1]}_{quality}.zip"
        zip_path = Path(task_dir) / zip_name

        try:
            yield event.plain_result(f"📦 正在下载并打包（{quality}），请稍候...")
            data = await self._call(
                "gallery",
                {
                    "href": href,
                    "fetch_all_previews": True,
                    "max_previews": self._zip_limit(),
                    "thumb_size": (1 if quality == "low" else 0),
                    "resolve_preview_images": False,
                    "resolve_image_candidates": (quality == "high"),
                },
            )
            candidates = list(data.get("image_candidates", []) or [])
            views = list(data.get("view_hrefs", []) or [])
            host_fail: dict[str, int] = {}
            t0 = time.time()
            connector = aiohttp.TCPConnector(limit=16, limit_per_host=4, ttl_dns_cache=300, ssl=False, enable_cleanup_closed=True)
            async with aiohttp.ClientSession(connector=connector, trust_env=True) as dl_sess:
                if quality == "mid":
                    if not views:
                        yield event.plain_result("❌ 没有可下载的图片链接")
                        return

                    sem = asyncio.Semaphore(3)

                    async def _one_mid(idx: int, view: list[str]):
                        async with sem:
                            nl = None
                            seen: set[str] = set()
                            for _ in range(4):
                                try:
                                    cand = await self._resolve_view_candidate(view, nl, session=dl_sess)
                                except Exception:
                                    await asyncio.sleep(0.25)
                                    continue
                                url = str(cand.get("sample") or "").strip()
                                if not url:
                                    await asyncio.sleep(0.25)
                                    continue
                                ref = str(cand.get("referer") or "https://exhentai.org/").strip() or "https://exhentai.org/"
                                p1 = await self._download_image(url, temp_dir=img_dir, referer=ref, session=dl_sess, host_fail=host_fail)
                                if p1:
                                    return p1
                                k = str(cand.get("nl") or "").strip()
                                if not k or k in seen:
                                    break
                                seen.add(k)
                                nl = k
                            return None

                    paths = await asyncio.gather(
                        *[_one_mid(i + 1, v) for i, v in enumerate(views)],
                        return_exceptions=True,
                    )
                    ok: list[tuple[int, str]] = []
                    fail_idx: list[int] = []
                    for idx, p in enumerate(paths, 1):
                        if isinstance(p, str) and p:
                            ok.append((idx, p))
                        else:
                            fail_idx.append(idx)
                    total = len(views)
                else:
                    raw = list(data.get("thumbnails") or [])
                    if not raw:
                        info = data.get("info", {}) or {}
                        cov = info.get("cover")
                        if cov:
                            raw.append(cov)
                        raw.extend(list(data.get("thumbnails", []) or []))

                    previews = [str(u) for u in raw if isinstance(u, str) and u]

                    referers = ["https://exhentai.org/"] * len(previews)
                    if candidates:
                        previews = [str((it.get("origin") or it.get("best") or it.get("sample") or "")).strip() for it in candidates]
                        referers = [str((it.get("referer") or "https://exhentai.org/")).strip() for it in candidates]
                    elif quality == "low":
                        referers = [f"https://exhentai.org/g/{href[0]}/{href[1]}/"] * len(previews)
                    if not previews:
                        yield event.plain_result("❌ 没有可下载的图片链接")
                        return

                    sem = asyncio.Semaphore(2 if quality == "high" else 8)

                    async def _one(idx: int, url: str):
                        async with sem:
                            ref = referers[idx - 1] if idx - 1 < len(referers) else "https://exhentai.org/"
                            p1 = await self._download_image(url, temp_dir=img_dir, referer=ref, session=dl_sess, host_fail=host_fail)
                            if p1:
                                return p1
                            if candidates and idx - 1 < len(candidates):
                                s2 = str((candidates[idx - 1].get("sample") or "")).strip()
                                if s2 and s2 != url:
                                    p2 = await self._download_image(s2, temp_dir=img_dir, referer=ref, session=dl_sess, host_fail=host_fail)
                                    if p2:
                                        return p2
                            return None

                    paths = await asyncio.gather(
                        *[_one(i + 1, u) for i, u in enumerate(previews)],
                        return_exceptions=True,
                    )
                    ok: list[tuple[int, str]] = []
                    fail_idx: list[int] = []
                    for idx, p in enumerate(paths, 1):
                        if isinstance(p, str) and p:
                            ok.append((idx, p))
                        else:
                            fail_idx.append(idx)
                    total = len(previews)

            if not ok:
                yield event.plain_result("❌ 所有图片下载失败，无法打包")
                return

            elapsed = round(time.time() - t0, 2)
            top_hosts = sorted(host_fail.items(), key=lambda x: x[1], reverse=True)[:12]
            logger.info(f"[exapi_cosmos] exzip quality={quality} ok={len(ok)}/{total} fail={len(fail_idx)} elapsed={elapsed}s top_fail_hosts={top_hosts[:5]}")

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for idx, p in ok:
                    zf.write(p, arcname=f"{idx:04d}_{Path(p).name}")
                manifest = {
                    "href": href,
                    "quality": quality,
                    "total": total,
                    "downloaded": len(ok),
                    "failed": fail_idx,
                    "elapsed_sec": elapsed,
                    "host_fail_top": top_hosts,
                }
                zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

            yield event.chain_result([File(name=zip_name, file=str(zip_path))])
            yield event.plain_result(f"✅ 压缩包已发送，成功打包 {len(ok)}/{total} 张，失败 {len(fail_idx)} 张，用时 {elapsed} 秒")
        except Exception as e:
            logger.error(f"exzip失败: {e}")
            yield event.plain_result(f"❌ 压缩包生成/发送失败: {e}")
        finally:
            asyncio.create_task(self._cleanup_dir_later(task_dir, 300))



    @filter.regex(r"^/?eximg(?:\s+.*)?$")
    async def eximg(self, event: AstrMessageEvent):
        args = self._args(event)
        pending = self._get_zip_pending(event)
        if not pending:
            yield event.plain_result("❌ 当前没有待发送任务。请先用 /exi 查看详情。")
            return
        if not args:
            yield event.plain_result("📦 请选择发送方式与画质：\n压缩包: /exzip high|mid|low\n合并图: /eximg high|mid|low（每条最多20张，分批发送）\n取消: /exzip no")
            return

        op = args[0].lower().strip()
        if op in {"no", "n", "cancel", "取消", "否"}:
            self._clear_zip_pending(event)
            yield event.plain_result("🛑 已取消发送")
            return

        quality = None
        if op in {"yes", "y", "ok", "确认", "是", "high", "h", "原图"}:
            quality = "high"
        elif op in {"mid", "m", "medium", "resample", "sample", "中"}:
            quality = "mid"
        elif op in {"low", "l", "thumb", "缩略", "低"}:
            quality = "low"

        if not quality:
            yield event.plain_result("❌ 用法: /eximg high|mid|low|no")
            return

        href = pending.get("href")
        self._clear_zip_pending(event)
        if not isinstance(href, list) or len(href) < 2:
            yield event.plain_result("❌ 待发送数据无效，请重新 /exi")
            return

        task_dir = tempfile.mkdtemp(prefix="eximg_", dir=str(self._temp_root()))
        img_dir = Path(task_dir) / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        try:
            yield event.plain_result(f"🖼️ 正在下载并分批发送（{quality}），请稍候...")
            data = await self._call(
                "gallery",
                {
                    "href": href,
                    "fetch_all_previews": True,
                    "max_previews": self._zip_limit(),
                    "thumb_size": (1 if quality == "low" else 0),
                    "resolve_preview_images": False,
                    "resolve_image_candidates": (quality == "high"),
                },
            )
            candidates = list(data.get("image_candidates", []) or [])
            views = list(data.get("view_hrefs", []) or [])
            host_fail: dict[str, int] = {}
            t0 = time.time()
            connector = aiohttp.TCPConnector(limit=16, limit_per_host=4, ttl_dns_cache=300, ssl=False, enable_cleanup_closed=True)
            async with aiohttp.ClientSession(connector=connector, trust_env=True) as dl_sess:
                if quality == "mid":
                    if not views:
                        yield event.plain_result("❌ 没有可下载的图片链接")
                        return

                    sem = asyncio.Semaphore(3)

                    async def _one_mid(idx: int, view: list[str]):
                        async with sem:
                            nl = None
                            seen: set[str] = set()
                            for _ in range(4):
                                try:
                                    cand = await self._resolve_view_candidate(view, nl, session=dl_sess)
                                except Exception:
                                    await asyncio.sleep(0.25)
                                    continue
                                url = str(cand.get("sample") or "").strip()
                                if not url:
                                    await asyncio.sleep(0.25)
                                    continue
                                ref = str(cand.get("referer") or "https://exhentai.org/").strip() or "https://exhentai.org/"
                                p1 = await self._download_image(url, temp_dir=img_dir, referer=ref, session=dl_sess, host_fail=host_fail)
                                if p1:
                                    return p1
                                k = str(cand.get("nl") or "").strip()
                                if not k or k in seen:
                                    break
                                seen.add(k)
                                nl = k
                            return None

                    paths = await asyncio.gather(
                        *[_one_mid(i + 1, v) for i, v in enumerate(views)],
                        return_exceptions=True,
                    )
                    ok: list[tuple[int, str]] = []
                    fail_idx: list[int] = []
                    for idx, p in enumerate(paths, 1):
                        if isinstance(p, str) and p:
                            ok.append((idx, p))
                        else:
                            fail_idx.append(idx)
                    total = len(views)
                else:
                    raw = list(data.get("thumbnails") or [])
                    if not raw:
                        info = data.get("info", {}) or {}
                        cov = info.get("cover")
                        if cov:
                            raw.append(cov)
                        raw.extend(list(data.get("thumbnails", []) or []))

                    previews = [str(u) for u in raw if isinstance(u, str) and u]

                    referers = ["https://exhentai.org/"] * len(previews)
                    if candidates:
                        previews = [str((it.get("origin") or it.get("best") or it.get("sample") or "")).strip() for it in candidates]
                        referers = [str((it.get("referer") or "https://exhentai.org/")).strip() for it in candidates]
                    elif quality == "low":
                        referers = [f"https://exhentai.org/g/{href[0]}/{href[1]}/"] * len(previews)
                    if not previews:
                        yield event.plain_result("❌ 没有可下载的图片链接")
                        return

                    sem = asyncio.Semaphore(2 if quality == "high" else 8)

                    async def _one(idx: int, url: str):
                        async with sem:
                            ref = referers[idx - 1] if idx - 1 < len(referers) else "https://exhentai.org/"
                            p1 = await self._download_image(url, temp_dir=img_dir, referer=ref, session=dl_sess, host_fail=host_fail)
                            if p1:
                                return p1
                            if candidates and idx - 1 < len(candidates):
                                s2 = str((candidates[idx - 1].get("sample") or "")).strip()
                                if s2 and s2 != url:
                                    p2 = await self._download_image(s2, temp_dir=img_dir, referer=ref, session=dl_sess, host_fail=host_fail)
                                    if p2:
                                        return p2
                            return None

                    paths = await asyncio.gather(
                        *[_one(i + 1, u) for i, u in enumerate(previews)],
                        return_exceptions=True,
                    )
                    ok: list[tuple[int, str]] = []
                    fail_idx: list[int] = []
                    for idx, p in enumerate(paths, 1):
                        if isinstance(p, str) and p:
                            ok.append((idx, p))
                        else:
                            fail_idx.append(idx)
                    total = len(previews)

            if not ok:
                yield event.plain_result("❌ 所有图片下载失败，无法发送")
                return

            elapsed = round(time.time() - t0, 2)
            top_hosts = sorted(host_fail.items(), key=lambda x: x[1], reverse=True)[:12]
            logger.info(f"[exapi_cosmos] eximg quality={quality} ok={len(ok)}/{total} fail={len(fail_idx)} elapsed={elapsed}s top_fail_hosts={top_hosts[:5]}")

            async for r in self._send_images_nodes_batched(event, ok, total, len(fail_idx), 20):
                yield r
            yield event.plain_result(f"✅ 合并消息发送完成，成功 {len(ok)}/{total} 张，失败 {len(fail_idx)} 张，用时 {elapsed} 秒")
        except Exception as e:
            logger.error(f"eximg失败: {e}")
            yield event.plain_result(f"❌ 合并消息发送失败: {e}")
        finally:
            asyncio.create_task(self._cleanup_dir_later(task_dir, 300))
    async def _mid_retry(self, one, previews, referers, views, fail_idx, ok):
        nl_map = {}
        nl_seen = {}
        last_fail = 10**9
        for _round in range(5):
            if not fail_idx:
                break
            todo = []
            order = []
            for i in fail_idx:
                if i - 1 >= len(views):
                    continue
                v = views[i - 1]
                if not isinstance(v, list) or len(v) < 2:
                    continue
                k = nl_map.get(i)
                todo.append([v[0], v[1], k] if k else [v[0], v[1]])
                order.append(i)
            if not todo:
                break
            fresh = await self._call("resolve_views", {"views": todo})
            cand = list(fresh.get("candidates", []) or [])
            new_key = False
            for j, i in enumerate(order):
                if j >= len(cand):
                    continue
                c = cand[j] or {}
                u = str((c.get("sample") or c.get("best") or "")).strip()
                if u and i - 1 < len(previews):
                    previews[i - 1] = u
                r = str((c.get("referer") or "")).strip()
                if r and i - 1 < len(referers):
                    referers[i - 1] = r
                k = str((c.get("nl") or "")).strip()
                if k:
                    s = nl_seen.setdefault(i, set())
                    if k not in s:
                        s.add(k)
                        nl_map[i] = k
                        new_key = True
            retry = await asyncio.gather(*[one(i, previews[i - 1]) for i in fail_idx], return_exceptions=True)
            nf = []
            for i, pth in zip(fail_idx, retry):
                if isinstance(pth, str) and pth:
                    ok.append((i, pth))
                else:
                    nf.append(i)
            fail_idx = nf
            ok.sort(key=lambda x: x[0])
            if len(fail_idx) >= last_fail and not new_key:
                break
            last_fail = len(fail_idx)
        return ok, fail_idx
