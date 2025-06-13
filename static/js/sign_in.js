document.getElementById("signupForm").addEventListener("submit", function(e) {
    e.preventDefault();

    const formData = new FormData(this);
    const password = formData.get("password");
    const confirmPassword = formData.get("confirm_password");

    if (password !== confirmPassword) {
        alert("비밀번호가 일치하지 않습니다.");
        return;
    }

    fetch("/sign-in", {
        method: "POST",
        body: formData
    })
    .then(response => {
        if (response.status === 200) {
            alert("회원가입 성공!");
            window.location.href = "/login";
        } else if (response.status === 409) {
            alert("이미 사용 중인 아이디입니다.");
        } else {
            alert("회원가입 실패. 다시 시도해주세요.");
        }
    })
    .catch(error => {
        console.error("Error:", error);
        alert("오류 발생");
    });
});
