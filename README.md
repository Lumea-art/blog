# 工程笔记

记录给 [GPTQModel](https://github.com/ModelCloud/GPTQModel) 提交内核与修复时踩过的坑：
为什么这么改、试过什么不work、数字是怎么量出来的。

## 文章

- [01 · 把一个 vLLM 内核搬进 GPTQModel：AllSpark W8A16 移植记](posts/01-allspark-kernel-port.md)

## 为什么写这些

内核 PR 的正文放不下过程：失败的尝试、被推翻的假设、量不出来的收益。
但这些恰恰是下一个做同样事情的人最需要的。

每篇都尽量写清三件事：

1. **动机** —— 解决什么问题，不解决的代价是什么
2. **原理** —— 为什么这么设计，换个做法会怎样
3. **证据** —— 数字从哪来，什么条件下量的，什么没能验证

## 许可

正文采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，
代码片段采用 MIT。
