// Protected functional behavior; editable pages supply accessible markup.
document.addEventListener('change', event => {
  if (event.target.id !== 'schedule-filter') return;
  const category=event.target.value;
  let count=0;
  for (const card of document.querySelectorAll('[data-event]')) {
    card.hidden=category!=='all' && card.dataset.category!==category;
    if (!card.hidden) count++;
  }
  const empty=document.getElementById('schedule-empty');if (empty) empty.hidden=count!==0;
});
document.addEventListener('submit', event => {
  const kind=event.target.id==='ticket-form'?'ticket':event.target.id==='volunteer-form'?'volunteer':null;
  if (!kind) return;
  event.preventDefault();
  const email=document.getElementById(kind+'-email'), status=document.getElementById(kind+'-status');
  if (!email || !email.checkValidity() || !status) return;
  status.textContent=kind==='ticket'?"You're on the list — we'll email ticket information.":"Thanks — we'll send volunteer information.";
  event.target.dataset.submitted='true';
});
