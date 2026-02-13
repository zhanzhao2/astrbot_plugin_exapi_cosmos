<div align="center">
  <img src="logo.png" alt="exApi-Cosmos Logo" width="160" />
</div>

# <div align="center">exApi-Cosmos</div>

<div align="center">
  <strong>AstrBot exHentai 搜索、预览、下载与发送插件</strong>
</div>

<br>
<div align="center">
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/VERSION-v0.2.5-6C5CE7?style=for-the-badge" alt="Version"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-009688?style=for-the-badge" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/PYTHON-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
</div>

<div align="center">
  <a href="https://github.com/AstrBotDevs/AstrBot"><img src="https://img.shields.io/badge/AstrBot-Compatible-00B894?style=for-the-badge" alt="AstrBot"></a>
  <a href="https://github.com/botuniverse/onebot-11"><img src="https://img.shields.io/badge/OneBotv11-NapCat-FF5722?style=for-the-badge" alt="OneBot"></a>
  <a href="https://github.com/zhanzhao2/astrbot_plugin_exapi_cosmos/releases"><img src="https://img.shields.io/badge/Releases-Public-2196F3?style=for-the-badge" alt="Releases"></a>
</div>

## 介绍

exApi-Cosmos 基于 exApi + AstrBot，提供 exHentai 的首页浏览、关键词搜索、高级搜索、详情预览、分批合并图发送与压缩包发送能力。

> [!IMPORTANT]
> **如果 AstrBot 与 NapCat 分离部署，必须配置共享卷，否则文件/图片发送会失败。**
> 本插件优先使用 `/shared_files/exapi_cosmos` 作为临时目录，发送链路依赖两端同时可访问该路径。

## 功能特性

### 核心功能
- **首页与搜索**：`/exhome`、`/exs`、`/exa`
- **详情索引**：`/exi 1` 直接基于最近搜索结果查看详情（无需手填 gid/token）
- **预览图**：详情后自动发送预览图（合并消息）
- **发送模式**：支持压缩包发送(`/exzip`)与图片合并消息发送(`/eximg`)
- **多档画质**：`high` 原图、`mid` 中质重采样、`low` 快速大缩略图

### 下载稳定性策略（已实现）
- 分块流式下载（`.part` 完成后原子替换）
- Content-Length 校验，短包自动判失败重试
- 主机失败计数与降载
- H@H 链接在失败场景自动尝试协议翻转（http/https）
- 下载任务结束后延迟清理 + 每小时孤儿目录回收

## 安装与依赖

### 1) 放置插件目录
将本仓库放入 AstrBot 插件目录：`/AstrBot/data/plugins/astrbot_plugin_exapi_cosmos`。

### 2) 安装 Python 依赖
```bash
pip install -r requirements.txt
```

### 3) 安装 Node 依赖（必须）
```bash
cd node/exapi_lib
npm install --omit=dev
```

### 4) 重启 AstrBot
确保插件加载成功。

### 5) 配置 Cookie
在插件配置中填写：`ipb_member_id`、`ipb_pass_hash`、`igneous`。

## 命令列表

| 命令 | 说明 |
|---|---|
| `/exstatus` | 检查配置、代理与连通性 |
| `/exhome [页码]` | 浏览首页内容 |
| `/exs <关键词> [页码]` | 关键词搜索 |
| `/exa key=value ...` | 高级搜索（或 JSON 参数） |
| `/exi <序号\|gid/token\|URL>` | 查看详情（支持 `/exi 1`） |
| `/exzip high\|mid\|low\|no` | 发送压缩包 |
| `/eximg high\|mid\|low\|no` | 发送合并图（最多 20 张/条，分批） |

### 典型流程
1. `/exs 关键词`
2. `/exi 1`（或 `/exi 2`）
3. 按提示二选一：`/exzip high|mid|low` 或 `/eximg high|mid|low`

## 分离部署（AstrBot / NapCat）必读

> [!CAUTION]
> **AstrBot 与 NapCat 分离部署时，必须配置共享卷，且两端挂载到相同路径（推荐 `/shared_files`）。**
> 否则会出现：文件发不出、图片链失败、仅文本返回、`File` 组件发送失败等问题。

### 推荐挂载
```yaml
services:
  astrbot:
    volumes:
      - ./shared_files:/shared_files
  napcat:
    volumes:
      - ./shared_files:/shared_files
```

### 本插件临时目录策略
- 首选：`/shared_files/exapi_cosmos`
- 备选：`<plugin_dir>/_tmp`（仅单容器/同文件系统场景）
- 任务目录前缀：`exzip_*`、`eximg_*`、`exprev_*`

## 清理策略

- 任务结束后延迟清理：默认 300 秒（zip/img）
- 预览图链路临时文件：90~120 秒清理
- 孤儿目录兜底清理：每小时执行一次（默认清理 6 小时前残留）
- NapCat 自身缓存（如 `/app/.config/QQ/NapCat/temp`）不由本插件直接管理

## 常见问题（FAQ）

**Q1：为什么日志有很多 SSL/EOF 报错，但最终还能发成功？**
- H@H 节点质量参差不齐，单节点失败常见；插件会换源重试并继续汇总可用结果。

**Q2：mid 为什么有时比 high 更容易失败？**
- mid 依赖页面解析得到 sample 链接，链路更长、节点波动更敏感。

**Q3：如何提高成功率？**
- 开启稳定代理；降低并发；优先 high；避开网络高峰；确保 cookie 有效。

## 配置项说明

| 配置项 | 说明 |
|---|---|
| `ipb_member_id` | exHentai Cookie 字段 |
| `ipb_pass_hash` | exHentai Cookie 字段 |
| `igneous` | exHentai Cookie 字段 |
| `use_proxy` / `proxy_url` | 可选代理配置 |
| `preview_limit` | 详情预览图上限 |
| `zip_image_limit` | 压缩包/合并图下载上限 |
| `search_page_size` | 搜索结果每页显示数 |
| `request_timeout` | 请求超时（秒） |
| `tls_verify` | TLS 证书校验（默认开启，建议保持） |
| `trust_env` | aiohttp 是否读取环境变量代理（默认关闭） |
| `node_bin` | Node 可执行文件名/路径 |
| `gallery_dl_bin` | gallery-dl 可执行文件名/路径 |
| `gallery_dl_timeout_sec` | gallery-dl 单次下载超时（秒） |
| `max_redirects` | fullimg 跳转最大次数（默认 5） |
| `cache_ttl_sec` | 会话缓存 TTL（默认 3600 秒） |
| `cache_max_sessions` | 会话缓存最大条数（默认 200） |
| `force_proxy_for_covers` | 封面/预览图下载是否强制代理（默认开启） |
| `hath_use_proxy` | H@H 下载是否使用代理（默认关闭） |

## 目录结构
- `main.py`：命令路由、会话状态、消息发送与调用编排
- `ex_parser.py`：HTML/URL 纯解析函数（extract/split/normalize）
- `downloader.py`：图片下载、gallery-dl 集成、mid 链路解析与重试
- `_conf_schema.json`：配置 schema
- `node/bridge.js`：Python ↔ Node 桥接
- `node/exapi_lib/`：页面解析和下载辅助库

## 致谢

- 感谢开源项目 **gallery-dl**(https://github.com/mikf/gallery-dl)，本插件在 H@H 图片下载链路中引入其能力以提升稳定性与兼容性。
- 感谢 **exApi / Ehviewer** 社区的开源实践与思路参考。

## 免责声明
本项目仅用于技术研究与学习交流，请遵守相关法律法规与站点条款。
