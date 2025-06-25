document.addEventListener("DOMContentLoaded", function () {
  const canvas = document.getElementById("scoreGauge");
  if (!canvas) return;

  const score = parseInt(canvas.dataset.score) || 0;

  // 점수 구간에 따른 색상 설정
  let color;
  if (score >= 90) {
    color = "#4CAF50"; // 초록
  } else if (score >= 70) {
    color = "#FFCA28"; // 노랑
  } else if (score >= 50) {
    color = "#FF7043"; // 주황
  } else {
    color = "#F44336"; // 빨강
  }

  const ctx = canvas.getContext("2d");
  new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["정돈 점수", "남은 점수"],
      datasets: [
        {
          data: [score, 100 - score],
          backgroundColor: [color, "#e0e0e0"],
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
