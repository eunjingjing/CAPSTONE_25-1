document.addEventListener("DOMContentLoaded", () => {
  const logoutBtn = document.getElementById("logoutBtn");

  if (logoutBtn) {
    logoutBtn.addEventListener("click", async (e) => {
      e.preventDefault();

      const csrfToken = document.querySelector('input[name="csrf_token"]')?.value;

      try {
        const response = await fetch("/logout", {
          method: "POST",
          headers: {
            "X-CSRFToken": csrfToken,
          }
        });

        const result = await response.json();
        alert(result.message);

        if (result.success) {
          window.location.href = "/";
        }

      } catch (error) {
        console.error("Logout error:", error);
        alert("로그아웃 중 오류가 발생했습니다.");
      }
    });
  }
});
