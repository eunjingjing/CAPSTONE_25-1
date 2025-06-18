document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("resetPasswordForm");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData(form);
    const token = window.location.pathname.split("/").pop();

    const response = await fetch(`/reset_password/${token}`, {
      method: "POST",
      body: formData,
    });

    const result = await response.json();

    alert(result.message);

    if (result.success) {
      window.location.href = "/login";
    }
  });
});
