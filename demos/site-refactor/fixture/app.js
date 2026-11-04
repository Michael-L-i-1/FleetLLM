const search = document.querySelector('#search');
const workshops = [...document.querySelectorAll('[data-workshop]')];
function filterWorkshops() {
  const query = search.value.toLowerCase().trim();
  workshops.forEach(card => { card.hidden = !card.textContent.toLowerCase().includes(query); });
  const visible = workshops.filter(card => !card.hidden).length;
  document.querySelector('#count').textContent = `${visible} workshop${visible === 1 ? '' : 's'}`;
}
search.addEventListener('input', filterWorkshops);
filterWorkshops();
document.querySelector('#signup-form').addEventListener('submit', event => {
  event.preventDefault();
  document.querySelector('#signup-status').textContent = "You're on the list!";
});
