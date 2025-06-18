document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("signupForm");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData(form);
    const password = formData.get("password");
    const confirmPassword = formData.get("confirm_password");

    if (password !== confirmPassword) {
      alert("비밀번호가 일치하지 않습니다.");
      return;
    }

    try {
      const response = await fetch("/sign-in", {
        method: "POST",
        body: formData
      });

      const result = await response.json();
      alert(result.message);

      if (result.success) {
        window.location.href = "/login";
      }

    } catch (error) {
      console.error("Error:", error);
      alert("오류가 발생했습니다.");
    }
  });
});
