# YongCHN 技术笔记

一个基于 Jekyll 与 GitHub Pages 的中文技术博客模板，适合发布原创技术文章、研究笔记和英文技术博客译文。

## 本地预览

1. 安装 Ruby 与 Bundler。
2. 运行 `bundle install`。
3. 运行 `bundle exec jekyll serve`。
4. 访问 `http://127.0.0.1:4000`。

## 发布文章

在 `_posts` 目录创建 `YYYY-MM-DD-slug.md`，并填写：

```yaml
---
layout: post
title: "文章标题"
description: "一句话摘要"
date: 2026-08-24 10:00:00 +0800
categories: [人工智能, 大语言模型]
tags: [LLM, Transformer]
author: Yong
translation: false
---
```

发布译文时，将 `translation` 改为 `true`，并补充原文信息：

```yaml
original_title: "Original article title"
original_author: "Original author"
original_url: "https://example.com/original-post"
```

推送到 `main` 分支后，GitHub Pages 会自动构建站点。
