# exapi_lib（供 AstrBot 插件内部使用）

本目录用于 `astrbot_plugin_exapi_cosmos` 的 Node 侧解析与抓取能力，
在本插件中通过 `node/bridge.js` 被调用。

## 来源说明
- 当前使用的代码来源：`https://github.com/zhanzhao2/exhentai-api`。
- 旧文档中的 `npm install exapi` 属于历史描述。
- 本插件不依赖线上 npm 包，而是直接使用仓库内本地源码。

## 在本插件中的安装方式
```bash
cd node/exapi_lib
npm install --omit=dev
```

## 说明
- 这是插件内嵌子模块文档，不是独立发布说明页。
- 插件使用说明请查看仓库根目录 `README.md`。
