const cards = document.querySelectorAll('.record-card');
const modal = document.getElementById('detailModal');
const modalImage = document.getElementById('modalImage');
const modalDate = document.getElementById('modalDate');
const modalScore = document.getElementById('modalScore');
const modalComment = document.getElementById('modalComment');
const closeModal = document.getElementById('closeModal');

cards.forEach(card => {
  card.addEventListener('click', () => {
    modalImage.src = card.dataset.img;
    modalDate.textContent = card.dataset.date;
    modalScore.textContent = card.dataset.score + "점";
    modalComment.textContent = card.dataset.comment;
    modal.style.display = 'flex';
  });
});

closeModal.addEventListener('click', () => {
  modal.style.display = 'none';
});

window.addEventListener('click', (e) => {
  if (e.target === modal) {
    modal.style.display = 'none';
  }
}); 