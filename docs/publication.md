# 开源发布说明

发布日期：2026-11-06。

仓库：[S4Plus/PROSE](https://github.com/S4Plus/PROSE)，默认分支为 `main`。

## 代码来源

本次发布导入整理后的 PROSE 项目快照。项目最初基于 [sliu76879-svg/Cupid_new](https://github.com/sliu76879-svg/Cupid_new) 的 `cupid` 分支；原始 CUPID 的项目介绍、作者、引用信息和上游致谢保存在 [历史 README](archive/CUPID_original.md)。

框架代码、不同实验实现、研究笔记和原有许可证随项目保留。项目根目录 [LICENSE](../LICENSE) 以及各框架目录内的许可证继续适用于各自内容。此次发布保留文件内容和来源说明，不导入来源仓库的历史 Git 对象。

## 发布范围

- 包含整理后的训练、评估、影响计算、数据筛选和再训练代码，以及环境配置、复现指南和轻量验证工具。
- 保留本地快照中实际存在的框架和方法版本；数据集、实验权重、缓存和运行日志由用户另行准备。
- README 的克隆地址更新为本仓库，首次发布分支为 `main`。
- 对 Windows 大小写冲突按实际文件名去重；OpenPI 框架说明与原实验笔记分别位于 `openpi/README.md`、`openpi/PROJECT_NOTES.md`。

## 验证范围

发布前重新运行仓库结构检查和复现入口测试。已有合成数据与脚本语法验证记录见 [validation.md](validation.md)。完整训练、仿真成功率和论文数值仍需根据 [复现指南](reproduction.md) 在相应环境、数据及模型权重下验证。
