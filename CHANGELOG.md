# Changelog

## v0.2.3 - 2026-02-12
- H@H 下载策略调整为仅直连 gallery-dl（不走代理）
- 封面/海报下载强制走代理
- 下载器重构：exzip/eximg 共用下载入口，mid rescue 复用主流程

## v0.2.2 - 2026-02-12
- 下载策略：H@H 强制直连 gallery-dl；封面/海报强制走代理
- 修复：showpage API 参数类型（gid/page）
- 重构：拆分 parser/downloader/main 并删除冗余重复下载逻辑
- 优化：mid 下载 rescue 复用主流程，减少重复代码

## v0.2.1 - 2026-02-11
- 新增 logo 并重构 README（对齐 JM-Cosmos 风格）
- 明确说明：AstrBot 与 NapCat 分离部署时，必须配置共享卷才能发送文件
- 补充分离部署示例、命令说明、常见问题与排障建议
- 发布首个 GitHub Release（Public）

## v0.2.0 - 2026-02-11
- 新增 `/eximg high|mid|low`：按最多 20 张/条分批发送合并消息
- `/exi` 后增加发送方式选择：`/exzip` 或 `/eximg`
- 优化下载层：分块写入、重试、失败主机统计、协议回退
- 增加压缩包质量档位：high/mid/low
- 增加任务目录延迟清理 + 每小时孤儿目录清理

## v0.1.0 - 2026-02-10
- 初版：首页、搜索、高级搜索、详情与预览
