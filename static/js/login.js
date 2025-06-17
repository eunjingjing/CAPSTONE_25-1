document.getElementById("loginForm").addEventListener("submit", function(e) {
    e.preventDefault();

    const formData = new FormData(this);

    fetch("/login", {
        method: "POST",
        body: formData
    })
    .then(response => {
        if (response.ok) {
            alert("로그인 성공!");
            window.location.href = "/";
        } else {
            alert("아이디 또는 비밀번호가 올바르지 않습니다.");
        }
    })
    .catch(error => {
        console.error("Error:", error);
        alert("오류 발생");
    });
});
