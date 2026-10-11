# 邮件发送与重试

云端每天按北京时间 11:23、15:23、19:23、23:23 尝试运行。不需要本地进程。GitHub cron 是尽力调度，不能承诺任务绝不漏触发。

## 2026-10-11 发现的平台阻断

Actions 页面明确提示该 fork 因 GitHub Actions 使用规模暂停运行，需要维护者重新启用。API 同时显示仓库权限 `enabled=true`、工作流 `active`，但手动触发邮件和 CI 均返回 HTTP 422 “Actions has been disabled for this repository”。再次设置仓库权限和调用工作流 enable API 没有解除这一限制。

维护者已在 Actions 页面重新启用。2026-10-11 北京时间 15:57:36，运行 [38122987673](https://github.com/suchencsw/zotero-arxiv-daily/actions/runs/38122987673) 的 SMTP 回执确认接受了 10 月 9 日历史批次补发邮件：100 篇候选，30 篇输出，本次读取的 Zotero 去重参考库 568 篇。HTML 产物包含中文 TLDR，并明确标注历史批次。SMTP 接受不代表收件箱已收到，仍需用户检查收件箱或垃圾箱。

同批次再次触发的运行 [38123171724](https://github.com/suchencsw/zotero-arxiv-daily/actions/runs/38123171724) 正确读取上述回执并跳过发送，没有重复联系 SMTP。云端 CI [38123123796](https://github.com/suchencsw/zotero-arxiv-daily/actions/runs/38123123796) 全部 120 项测试通过。

守卫不再查看任务的绿色状态，而是读取 `email-smtp-receipt` 回执：它只在 SMTP 接受消息后产生。记录包含北京时间日期、论文数量、候选数量与 Zotero 参考库数量，不包含账号、密码或 API 密钥。即使后续清理步骤失败，已有发送回执仍能阻止重复发送。

- 有论文且发送成功：当天后续正常尝试跳过。
- 无新公告：发送一次中文状态邮件；后续仍抓取公告，有新论文时允许再发，重复空状态不再发。
- 网络、Zotero、模型或 SMTP 失败：失败可见，无发送回执，后续尝试不会被当成已经发送。
- 验证模式：`verify_only=true`，不联系 SMTP，不产生发送回执。

无发送回执的任务不等于邮件已发送；SMTP 接受也不等于收件箱已收到。只能查看收件箱进一步确认最终投递。若执行器恰好在 SMTP 接受之后、回执上传之前被终止，仍可能发生重发；本机制不宣称跨 SMTP 与 GitHub 的事务性 exactly-once 保证。

## 手动补发

Actions → Send emails daily → Run workflow，`recovery_batch` 填写已保存批次，如 `2026-10-09`。此批次来自已核验的真实 arXiv 公告公共元数据，邮件主题和正文均标注“历史批次补发”，按本次新读取的 Zotero 库和现有邮件排序算法筛选，最多 100 候选、30 输出，不冒充今日新论文。该历史补发回执不会挡住当天正常新公告推送。

同批次同北京时间日期已补发成功时默认跳过。只有明确要重复发送时才启用 `force_send`。

调试产物包括渲染的 HTML 和 SMTP 回执，保存在每次 Actions 的 Artifacts 中。回执保留 30 天，HTML 保留 7 天。
