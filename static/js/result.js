document.addEventListener("DOMContentLoaded", function () {
  const canvas = document.getElementById("scoreGauge");
  if (!canvas) return;

  const score = parseInt(canvas.dataset.score) || 0;

  const ctx = canvas.getContext("2d");
  new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["정돈 점수", "남은 점수"],
      datasets: [
        {
          data: [score, 100 - score],
          backgroundColor: ["#4CAF50", "#e0e0e0"],
          borderWidth: 0,
        },
      ],
    },
    options: {
      rotation: -90,
      circumference: 180,
      cutout: "70%",
      plugins: {
        legend: { display: false },
        tooltip: { enabled: false },
      },
    },
  });
});
