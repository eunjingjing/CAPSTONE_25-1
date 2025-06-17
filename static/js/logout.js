function handleLogout() {
    fetch('/logout')
    .then(response => {
        if (response.ok) {
            alert("로그아웃 성공!");
            window.location.href = "/";
        } else {
            alert("로그아웃 실패!");
        }
    })
    .catch(error => {
        console.error("Error:", error);
        alert("오류 발생");
    });
}
