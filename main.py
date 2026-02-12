from __future__ import annotations

import asyncio
import socket
import aiohttp
import json
import os
import re
import shlex
import sys
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

_plugin_dir = Path(__file__).parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))
from downloader import ExDownloader

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
from astrbot.api.message_components import File, Image, Node, Nodes, Plain

@register("exapi_cosmos", "zhanzhao2", "exHentai 搜索插件（基于 exApi）", "0.2.4", "https://github.com/zhanzhao2/astrbot_plugin_exapi_cosmos")
class ExApiCosmosPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig = None):
        super().__init__(context)
        self.config = config or {}
        self.bridge = Path(__file__).parent / "node" / "bridge.js"
        self._last_items: dict[str, dict[str, Any]] = {}
        self._zip_pending: dict[str, dict[str, Any]] = {}
        self.downloader = ExDownloader(
            get_proxy=self._proxy,
            get_cookie_header=self._cookie_header,
            call_bridge=self._call,
            get_temp_root=self._temp_root,
            logger=logger,
            get_tls_verify=self._tls_verify,
            get_trust_env=self._trust_env,
            get_max_redirects=self._max_redirects,
            get_hath_use_proxy=self._hath_use_proxy,
        )
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

    def _bridge_proxy(self) -> str | None:
        p = self._proxy()
        if not p:
            return None
        return p if str(p).lower().startswith("socks") else None

    def _page_size(self) -> int:
        n = int(self.config.get("search_page_size", 5) or 5)
        return max(1, min(n, 20))

    def _timeout(self) -> int:
        n = int(self.config.get("request_timeout", 90) or 90)
        return max(10, min(n, 300))

    def _tls_verify(self) -> bool:
        return bool(self.config.get("tls_verify", True))

    def _trust_env(self) -> bool:
        return bool(self.config.get("trust_env", False))

    def _max_redirects(self) -> int:
        n = int(self.config.get("max_redirects", 5) or 5)
        return max(0, min(n, 10))

    def _cache_ttl_sec(self) -> int:
        n = int(self.config.get("cache_ttl_sec", 3600) or 3600)
        return max(60, min(n, 86400))

    def _cache_max_sessions(self) -> int:
        n = int(self.config.get("cache_max_sessions", 200) or 200)
        return max(20, min(n, 2000))

    def _force_proxy_for_covers(self) -> bool:
        return bool(self.config.get("force_proxy_for_covers", True))

    def _hath_use_proxy(self) -> bool:
        return bool(self.config.get("hath_use_proxy", False))

    def _cache_key(self, event: AstrMessageEvent) -> str:
        return f"{event.get_platform_id()}:{event.get_session_id()}"

    def _cache_gc(self):
        now = time.time()
        ttl = self._cache_ttl_sec()
        cap = self._cache_max_sessions()

        def _trim(store: dict[str, dict[str, Any]]):
            for k in list(store.keys()):
                v = store.get(k) or {}
                ts = float(v.get("ts", 0.0) or 0.0)
                if now - ts > ttl:
                    store.pop(k, None)
            if len(store) <= cap:
                return
            ordered = sorted(store.items(), key=lambda kv: float((kv[1] or {}).get("ts", 0.0) or 0.0), reverse=True)
            keep = {k for k, _ in ordered[:cap]}
            for k in list(store.keys()):
                if k not in keep:
                    store.pop(k, None)

        _trim(self._last_items)
        _trim(self._zip_pending)

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
                if p.is_dir() and p.name.startswith(("exzip_", "exprev_", "eximg_", "exsr_")) and now - p.stat().st_mtime > max_age_sec:
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

    def _eximg_batch_size(self) -> int:
        n = int(self.config.get("eximg_batch_size", 20) or 20)
        return max(3, min(n, 20))

    def _eximg_send_interval(self) -> float:
        n = float(self.config.get("eximg_send_interval_sec", 1.2) or 1.2)
        return max(0.3, min(n, 5.0))

    def _set_zip_pending(self, event: AstrMessageEvent, href: list[str]):
        self._cache_gc()
        self._zip_pending[self._cache_key(event)] = {
            "ts": time.time(),
            "href": [str(href[0]), str(href[1])],
        }

    def _get_zip_pending(self, event: AstrMessageEvent) -> dict[str, Any] | None:
        self._cache_gc()
        val = self._zip_pending.get(self._cache_key(event))
        if not isinstance(val, dict):
            return None
        href = val.get("href")
        if not isinstance(href, list) or len(href) < 2:
            return None
        return {"href": [str(href[0]), str(href[1])]}

    def _clear_zip_pending(self, event: AstrMessageEvent):
        self._zip_pending.pop(self._cache_key(event), None)

    def _save_last_items(self, event: AstrMessageEvent, items: list[dict[str, Any]]):
        self._cache_gc()
        rows: list[dict[str, Any]] = []
        for it in items[: self._page_size()]:
            h = it.get("href") or []
            if isinstance(h, list) and len(h) >= 2:
                rows.append({"href": [str(h[0]), str(h[1])], "title": str(it.get("title", "未知标题"))})
        self._last_items[self._cache_key(event)] = {
            "ts": time.time(),
            "items": rows,
        }

    def _pick_cached_href(self, event: AstrMessageEvent, idx: int) -> list[str] | None:
        self._cache_gc()
        entry = self._last_items.get(self._cache_key(event), {})
        arr = entry.get("items", []) if isinstance(entry, dict) else []
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


    async def _send_previews(self, event: AstrMessageEvent, previews: list[str]):
        """并发下载后，最终只发送一条合并消息。"""
        if not previews:
            return

        uin = str(event.get_self_id() or event.get_sender_id() or "0")
        tmp_dir = Path(tempfile.mkdtemp(prefix="exprev_", dir=str(self._temp_root())))

        sem = asyncio.Semaphore(2)
        cover_force_proxy = self._force_proxy_for_covers()
        cover_use_proxy = cover_force_proxy and bool(self._proxy())
        cover_verify_tls = self._tls_verify()
        cover_connector = aiohttp.TCPConnector(
            limit=16,
            limit_per_host=4,
            ttl_dns_cache=300,
            ssl=(False if not cover_verify_tls else None),
            enable_cleanup_closed=True,
            family=socket.AF_INET,
        )

        async with aiohttp.ClientSession(connector=cover_connector, trust_env=self._trust_env()) as cover_sess:
            async def _one(idx: int, url: str):
                async with sem:
                    p = await self.downloader.download_image(
                        url,
                        temp_dir=tmp_dir,
                        session=cover_sess,
                        force_proxy=cover_use_proxy,
                    )
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
        interval: float | None = None,
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
                await asyncio.sleep(interval if interval is not None else self._eximg_send_interval())

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
        body = {"action": action, "cookies": self._cookies(), "proxy": self._bridge_proxy(), **payload}
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

        stderr_text = err.decode("utf-8", errors="ignore").strip()
        if p.returncode not in (0, None):
            msg = stderr_text or f"bridge 进程退出码 {p.returncode}"
            raise RuntimeError(msg)

        text = out.decode("utf-8", errors="ignore").strip()
        if not text:
            msg = stderr_text[:300] if stderr_text else "bridge 无返回"
            raise RuntimeError(msg)

        data = None
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and ("ok" in obj or "data" in obj or "error" in obj):
                data = obj
                break
        if not isinstance(data, dict):
            raise RuntimeError("bridge 返回格式异常（未找到 JSON 结果）")

        if not data.get("ok", False):
            em = str(data.get("error", "") or "").strip()
            if not em and stderr_text:
                em = stderr_text[:300]
            raise RuntimeError(em or "未知错误")
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

    async def _send_search_with_cover(self, event: AstrMessageEvent, title: str, items: list[dict[str, Any]], page: int, pages: int, nxt: str, extra: str | None = None):
        shown = list(items[: self._page_size()])
        msg = self._fmt_list(title, items, page, pages, nxt)
        if extra:
            msg += "\n\n" + str(extra)[:360]
        if not shown:
            yield event.plain_result(msg)
            return

        tmp = Path(tempfile.mkdtemp(prefix="exsr_", dir=str(self._temp_root())))
        sem = asyncio.Semaphore(4)
        cover_force_proxy = self._force_proxy_for_covers()
        cover_use_proxy = cover_force_proxy and bool(self._proxy())
        cover_verify_tls = self._tls_verify()
        cover_connector = aiohttp.TCPConnector(
            limit=20,
            limit_per_host=6,
            ttl_dns_cache=300,
            ssl=(False if not cover_verify_tls else None),
            enable_cleanup_closed=True,
            family=socket.AF_INET,
        )

        try:
            async with aiohttp.ClientSession(connector=cover_connector, trust_env=self._trust_env()) as cover_sess:
                async def _one(idx: int, it: dict[str, Any]):
                    async with sem:
                        cover = str(it.get("cover", "")).strip()
                        if not cover:
                            return idx, None
                        h = it.get("href") or []
                        ref = "https://exhentai.org/"
                        if isinstance(h, list) and len(h) >= 2:
                            ref = f"https://exhentai.org/g/{h[0]}/{h[1]}/"
                        p1 = await self.downloader.download_image(
                            cover,
                            temp_dir=tmp,
                            referer=ref,
                            session=cover_sess,
                            fast=True,
                            force_proxy=cover_use_proxy,
                        )
                        return idx, p1

                rs = await asyncio.gather(*[_one(i + 1, it) for i, it in enumerate(shown)], return_exceptions=True)
            cover_map: dict[int, str] = {}
            for r in rs:
                if isinstance(r, tuple) and len(r) == 2 and isinstance(r[0], int) and isinstance(r[1], str) and r[1]:
                    cover_map[r[0]] = r[1]

            use_nodes = len(msg) > 700 or len(shown) >= 4
            if use_nodes:
                uin = str(event.get_self_id() or event.get_sender_id() or "0")
                head = [title, "━━━━━━━━━━━━━━━━━━━━━"]
                if pages and pages > 0:
                    head.append(f"📚 总页: {pages}")
                head += [f"📄 第 {page} 页", f"💡 下一页: {nxt}", "💡 详情: /exi <序号|gid/token>"]
                nodes = [Node(name="exApi-Cosmos", uin=uin, content=[Plain("\n".join(head))])]
                for i, it in enumerate(shown, 1):
                    h = it.get("href") or []
                    gid = f"{h[0]}/{h[1]}" if isinstance(h, list) and len(h) >= 2 else "N/A"
                    t = str(it.get("title", "未知标题")).replace("\n", " ").strip()[:52]
                    row = [Plain(f"{i}. 【{gid}】{t}")]
                    p = cover_map.get(i)
                    if p:
                        row.append(Image.fromFileSystem(p))
                    nodes.append(Node(name="exApi-Cosmos", uin=uin, content=row))
                if extra:
                    nodes.append(Node(name="exApi-Cosmos", uin=uin, content=[Plain(str(extra)[:360])]))
                yield event.chain_result([Nodes(nodes)])
            else:
                chain = [Plain(f"{title}\n━━━━━━━━━━━━━━━━━━━━━")]
                for i, it in enumerate(shown, 1):
                    h = it.get("href") or []
                    gid = f"{h[0]}/{h[1]}" if isinstance(h, list) and len(h) >= 2 else "N/A"
                    t = str(it.get("title", "未知标题")).replace("\n", " ").strip()[:52]
                    chain.append(Plain(f"\n{i}. 【{gid}】{t}"))
                    p = cover_map.get(i)
                    if p:
                        chain.append(Image.fromFileSystem(p))
                tail = []
                if pages and pages > 0:
                    tail.append(f"📚 总页: {pages}")
                tail += [f"📄 第 {page} 页", f"💡 下一页: {nxt}", "💡 详情: /exi <序号|gid/token>"]
                if extra:
                    tail += ["", str(extra)[:360]]
                chain.append(Plain("\n\n" + "\n".join(tail)))
                yield event.chain_result(chain)
        except Exception as e:
            logger.warning(f"搜索结果封面发送失败，回退纯文本: {e}")
            yield event.plain_result(msg)
        finally:
            asyncio.create_task(self._cleanup_dir_later(str(tmp), 180))

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
            "/eximg mid|low|no - 分批合并消息发图（最高mid，每条最多20张（可配置））\n"
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
            async for r in self._send_search_with_cover(event, f"🏠 首页 (第{now}页)", items, now, pages, f"/exhome {now+1}"):
                yield r
        except Exception as e:
            logger.error(f"exhome失败: {e}")
            yield event.plain_result(f"❌ 首页获取失败: {e}")

    @filter.regex(r"^/?(?:exs|ex)(?:\s+.*)?$")
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
            async for r in self._send_search_with_cover(event, f"🔍 搜索: {kw} (第{now}页)", items, now, pages, f"/exs {kw} {now+1}"):
                yield r
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
            async for r in self._send_search_with_cover(
                event,
                f"🧠 高级搜索 (第{now}页)",
                items,
                now,
                pages,
                f"/exa page={now+1} ...",
                "配置: " + json.dumps(cfg, ensure_ascii=False),
            ):
                yield r
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
                cov = info.get("cover")
                if cov:
                    raw.append(cov)

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
            yield event.plain_result("📦 请选择发送方式与画质：\n压缩包: /exzip high|mid|low\n合并图: /eximg mid|low（每条最多20张（可配置），分批发送）\n取消: /exzip no")
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
            yield event.plain_result("📦 请选择发送方式与画质：\n压缩包: /exzip high|mid|low\n合并图: /eximg mid|low（每条最多20张（可配置），分批发送）\n取消: /exzip no")
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
            t_gallery0 = time.perf_counter()
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
            gallery_elapsed = round(time.perf_counter() - t_gallery0, 3)
            logger.info(f"[exapi_cosmos][exzip-prof] stage=gallery quality={quality} views={len(views)} candidates={len(candidates)} elapsed={gallery_elapsed}s")
            host_fail: dict[str, int] = {}
            t0 = time.time()
            t_download0 = time.perf_counter()
            connector = aiohttp.TCPConnector(limit=48, limit_per_host=8, ttl_dns_cache=300, ssl=False, enable_cleanup_closed=True, family=socket.AF_INET)
            async with aiohttp.ClientSession(connector=connector, trust_env=True) as dl_sess:
                ok, fail_idx, total = await self.downloader.download_gallery_images(
                    quality=quality,
                    href=href,
                    data=data,
                    img_dir=img_dir,
                    session=dl_sess,
                    host_fail=host_fail,
                )
                if total <= 0:
                    yield event.plain_result("❌ 没有可下载的图片链接")
                    return

            dl_elapsed = round(time.perf_counter() - t_download0, 3)
            logger.info(f"[exapi_cosmos][exzip-prof] stage=download quality={quality} ok={len(ok)}/{total} fail={len(fail_idx)} elapsed={dl_elapsed}s")

            if not ok:
                yield event.plain_result("❌ 所有图片下载失败，无法打包")
                return

            elapsed = round(time.time() - t0, 2)
            top_hosts = sorted(host_fail.items(), key=lambda x: x[1], reverse=True)[:12]
            logger.info(f"[exapi_cosmos] exzip quality={quality} ok={len(ok)}/{total} fail={len(fail_idx)} elapsed={elapsed}s top_fail_hosts={top_hosts[:5]}")

            t_zip0 = time.perf_counter()
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

            zip_elapsed = round(time.perf_counter() - t_zip0, 3)
            logger.info(f"[exapi_cosmos][exzip-prof] stage=zip quality={quality} files={len(ok)} elapsed={zip_elapsed}s path={zip_path}")
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
            yield event.plain_result("📦 请选择发送方式与画质：\n压缩包: /exzip high|mid|low\n合并图: /eximg mid|low（每条最多20张（可配置），分批发送）\n取消: /exzip no")
            return

        op = args[0].lower().strip()
        if op in {"no", "n", "cancel", "取消", "否"}:
            self._clear_zip_pending(event)
            yield event.plain_result("🛑 已取消发送")
            return

        quality = None
        downgrade_high = False
        if op in {"yes", "y", "ok", "确认", "是", "high", "h", "原图"}:
            quality = "mid"
            downgrade_high = True
        elif op in {"mid", "m", "medium", "resample", "sample", "中"}:
            quality = "mid"
        elif op in {"low", "l", "thumb", "缩略", "低"}:
            quality = "low"

        if not quality:
            yield event.plain_result("❌ 用法: /eximg mid|low|no")
            return
        if downgrade_high:
            yield event.plain_result("ℹ️ 合并消息最高仅支持 mid，已自动降级。若需原图请使用 /exzip high")

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
            t_gallery0 = time.perf_counter()
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
            gallery_elapsed = round(time.perf_counter() - t_gallery0, 3)
            logger.info(f"[exapi_cosmos][eximg-prof] stage=gallery quality={quality} views={len(views)} candidates={len(candidates)} elapsed={gallery_elapsed}s")
            host_fail: dict[str, int] = {}
            t0 = time.time()
            t_download0 = time.perf_counter()
            connector = aiohttp.TCPConnector(limit=48, limit_per_host=8, ttl_dns_cache=300, ssl=False, enable_cleanup_closed=True, family=socket.AF_INET)
            async with aiohttp.ClientSession(connector=connector, trust_env=True) as dl_sess:
                ok, fail_idx, total = await self.downloader.download_gallery_images(
                    quality=quality,
                    href=href,
                    data=data,
                    img_dir=img_dir,
                    session=dl_sess,
                    host_fail=host_fail,
                )
                if total <= 0:
                    yield event.plain_result("❌ 没有可下载的图片链接")
                    return

            dl_elapsed = round(time.perf_counter() - t_download0, 3)
            logger.info(f"[exapi_cosmos][eximg-prof] stage=download quality={quality} ok={len(ok)}/{total} fail={len(fail_idx)} elapsed={dl_elapsed}s")

            if not ok:
                yield event.plain_result("❌ 所有图片下载失败，无法发送")
                return

            elapsed = round(time.time() - t0, 2)
            top_hosts = sorted(host_fail.items(), key=lambda x: x[1], reverse=True)[:12]
            logger.info(f"[exapi_cosmos] eximg quality={quality} ok={len(ok)}/{total} fail={len(fail_idx)} elapsed={elapsed}s top_fail_hosts={top_hosts[:5]}")
            safe_batch = self._eximg_batch_size()
            safe_interval = self._eximg_send_interval()
            if quality == "mid":
                safe_batch = 20
                safe_interval = max(safe_interval, 1.0)
            if total >= 20:
                safe_batch = 20
                safe_interval = max(safe_interval, 1.0)

            t_send0 = time.perf_counter()
            async for r in self._send_images_nodes_batched(event, ok, total, len(fail_idx), safe_batch, safe_interval):
                yield r
            send_elapsed = round(time.perf_counter() - t_send0, 3)
            batch_count = (len(ok) + safe_batch - 1) // safe_batch if safe_batch > 0 else 0
            logger.info(f"[exapi_cosmos][eximg-prof] stage=send quality={quality} batch_size={safe_batch} batches={batch_count} elapsed={send_elapsed}s")
            yield event.plain_result(f"✅ 合并消息发送完成，成功 {len(ok)}/{total} 张，失败 {len(fail_idx)} 张，用时 {elapsed} 秒")
        except Exception as e:
            logger.error(f"eximg失败: {e}")
            yield event.plain_result(f"❌ 合并消息发送失败: {e}")
        finally:
            asyncio.create_task(self._cleanup_dir_later(task_dir, 300))
