<div align="center">
  <h1>exApi-Cosmos</h1>
  <strong>AstrBot exHentai 搜索与下载插件</strong>
</div>

<div align="center">
  <img src="https://img.shields.io/badge/VERSION-v0.2.0-6C5CE7?style=for-the-badge" alt="Version">
  <img src="https://img.shields.io/badge/PYTHON-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/AstrBot-Compatible-00B894?style=for-the-badge" alt="AstrBot">
  <img src="https://img.shields.io/badge/Node.js-required-339933?style=for-the-badge&logo=nodedotjs&logoColor=white" alt="Node">
</div>

## 介绍

exApi-Cosmos 是一个基于 AstrBot 的 exHentai 插件，支持搜索、高级搜索、详情预览、图片合并消息发送与压缩包发送。

## 功能特性

- 关键词搜索：`/exs <关键词> [页码]`
- 高级搜索：`/exa key=value ...` 或 `/exa {json}`
- 详情查看：`/exi <序号|gid/token|URL>`
- 预览发送：详情后自动发送预览图（合并消息）
- 发送模式：`/exzip high|mid|low` 或 `/eximg high|mid|low`
- 合并图分批发送：每条最多 20 张
- 临时文件清理：任务结束延时清理 + 每小时孤儿清理

## 安装方法

1. 将本插件目录放入 AstrBot 的 `data/plugins/astrbot_plugin_exapi_cosmos`。
2. 安装 Python 依赖：

```bash
pip install -r requirements.txt
```

3. 安装 Node 依赖（必须）：

```bash
cd node/exapi_lib
npm install --omit=dev
```

4. 重启 AstrBot。

## 命令列表

- `/exstatus`：检查配置与连通性
- `/exhome [页码]`：浏览首页
- `/exs <关键词> [页码]`：关键词搜索
- `/exa key=value ...`：高级搜索
- `/exi <序号|gid/token|URL>`：查看详情与预览
- `/exzip high|mid|low|no`：发送压缩包（high 原图 / mid 中质 / low 快速）
- `/eximg high|mid|low|no`：发送合并消息（分批，每批最多 20 张）

## 配置项

请在插件配置中填写 exHentai Cookie：
- `ipb_member_id`
- `ipb_pass_hash`
- `igneous`

可选项：`use_proxy`、`proxy_url`、`preview_limit`、`zip_image_limit`。

## 目录结构

- `main.py`：AstrBot 插件主逻辑
- `_conf_schema.json`：配置 schema
- `node/bridge.js`：Python 与 Node 桥接
- `node/exapi_lib/`：exApi 解析与抓取逻辑

## 免责声明

本项目仅用于技术研究与学习交流，请遵守当地法律法规和网站条款。
