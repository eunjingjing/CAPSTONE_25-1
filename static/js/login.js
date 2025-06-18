document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("loginForm");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData(form);

    try {
      const response = await fetch("/login", {
        method: "POST",
        body: formData
      });

      const result = await response.json();

      alert(result.message);

      if (result.success) {
        window.location.href = "/";
      }

    } catch (error) {
      console.error("Login error:", error);
      alert("오류가 발생했습니다.");
    }
  });
});
