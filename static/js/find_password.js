document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("findPasswordForm");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData(form);

    const response = await fetch("/find-password", {
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
