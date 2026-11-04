// Protected behavior: models supply semantic markup and styling only.
document.addEventListener('input', event => {
  if (event.target.id !== 'menu-search') return;
  const needle = event.target.value.trim().toLowerCase();
  const items = [...document.querySelectorAll('[data-menu-item]')];
  let visible = 0;
  for (const item of items) {
    const match = (item.dataset.name || item.textContent).toLowerCase().includes(needle);
    item.hidden = !match;
    if (match) visible++;
  }
  const empty = document.getElementById('menu-empty');
  if (empty) empty.hidden = visible !== 0;
});
document.addEventListener('submit', event => {
  if (event.target.id !== 'newsletter-form') return;
  event.preventDefault();
  const input = document.getElementById('email');
  const status = document.getElementById('newsletter-status');
  if (!input || !status || !input.checkValidity()) return;
  status.textContent = 'Thanks — you are on the Common Ground list.';
  event.target.dataset.submitted = 'true';
});
