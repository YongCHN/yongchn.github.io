---
layout: page
title: 文章归档
permalink: /archive/
eyebrow: Archive
description: 按时间浏览全部原创文章、研究笔记与技术译文。
---

<div class="archive-list">
  {% assign posts_by_year = site.posts | group_by_exp: "post", "post.date | date: '%Y'" %}
  {% for year in posts_by_year %}
    <section class="archive-year">
      <h2>{{ year.name }}</h2>
      <div>
        {% for post in year.items %}
          <a class="archive-item" href="{{ post.url | relative_url }}">
            <time>{{ post.date | date: "%m.%d" }}</time>
            <span>{{ post.title }}</span>
            {% if post.translation %}<em>译文</em>{% else %}<em>原创</em>{% endif %}
          </a>
        {% endfor %}
      </div>
    </section>
  {% endfor %}
</div>
