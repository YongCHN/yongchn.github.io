const navToggle = document.querySelector('.nav-toggle');
const siteNav = document.querySelector('.site-nav');

if (navToggle && siteNav) {
  navToggle.addEventListener('click', () => {
    const isOpen = navToggle.getAttribute('aria-expanded') === 'true';
    navToggle.setAttribute('aria-expanded', String(!isOpen));
    navToggle.setAttribute('aria-label', isOpen ? '打开导航' : '关闭导航');
    siteNav.classList.toggle('is-open', !isOpen);
  });
}

const searchInput = document.querySelector('#post-search');
const filterButtons = [...document.querySelectorAll('.filter-pill')];
const posts = [...document.querySelectorAll('.searchable-post')];
const emptyState = document.querySelector('#empty-state');
let activeFilter = 'all';

function filterPosts() {
  const query = (searchInput?.value || '').trim().toLowerCase();
  let visibleCount = 0;

  posts.forEach((post) => {
    const searchable = `${post.dataset.title} ${post.dataset.categories} ${post.dataset.tags}`.toLowerCase();
    const kind = post.querySelector('.post-kind')?.textContent.trim();
    const matchesQuery = searchable.includes(query);
    const matchesFilter = activeFilter === 'all' || kind === activeFilter;
    const visible = matchesQuery && matchesFilter;
    post.hidden = !visible;
    if (visible) visibleCount += 1;
  });

  if (emptyState) emptyState.hidden = visibleCount !== 0;
}

searchInput?.addEventListener('input', filterPosts);
filterButtons.forEach((button) => {
  button.addEventListener('click', () => {
    activeFilter = button.dataset.filter;
    filterButtons.forEach((item) => item.classList.toggle('active', item === button));
    filterPosts();
  });
});
