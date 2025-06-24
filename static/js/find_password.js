document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("findPasswordForm");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData(form);

    try {
      const response = await fetch("/find-password", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      alert(result.message);

      if (result.success) {
        window.location.href = "/login";
      }
    } catch (err) {
      console.error("비밀번호 찾기 요청 실패:", err);
      alert("오류가 발생했습니다. 다시 시도해주세요.");
    }
  });
});
