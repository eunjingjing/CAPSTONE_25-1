document.addEventListener('DOMContentLoaded', () => {
    const uploadBox = document.querySelector('.upload-box');
    const fileInput = document.getElementById('imgUpload');

    // 기존 요소 캐시
    const icon = uploadBox.querySelector('.material-symbols-outlined');
    const texts = [...uploadBox.querySelectorAll('div, label')];

    // 미리보기 이미지 및 X버튼
    const previewWrapper = document.createElement('div');
    previewWrapper.style.position = 'relative';
    previewWrapper.style.display = 'none';
    uploadBox.appendChild(previewWrapper);

    const previewImg = document.createElement('img');
    previewImg.style.maxWidth = '30vw';
    previewImg.style.borderRadius = '10px';
    previewImg.style.marginTop = '10px';
    previewImg.style.boxShadow = '0 0 5px rgba(0,0,0,0.2)';
    previewWrapper.appendChild(previewImg);

    const deleteBtn = document.createElement('button');
    deleteBtn.textContent = '✕';
    deleteBtn.style.position = 'absolute';
    deleteBtn.style.top = '-10px';
    deleteBtn.style.right = '-10px';
    deleteBtn.style.background = '#f44336';
    deleteBtn.style.color = '#fff';
    deleteBtn.style.border = 'none';
    deleteBtn.style.borderRadius = '50%';
    deleteBtn.style.width = '25px';
    deleteBtn.style.height = '25px';
    deleteBtn.style.cursor = 'pointer';
    deleteBtn.style.fontSize = '16px';
    deleteBtn.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
    previewWrapper.appendChild(deleteBtn);

    const showPreview = file => {
    if (!file || !file.type.startsWith('image/')) return;

    // 텍스트, 아이콘 숨기기
    icon.style.display = 'none';
    texts.forEach(t => t.style.display = 'none');

    const reader = new FileReader();
    reader.onload = e => {
        previewImg.src = e.target.result;
        previewWrapper.style.display = 'block';
    };
    reader.readAsDataURL(file);
    };

    const resetUploadBox = () => {
    fileInput.value = '';
    previewWrapper.style.display = 'none';
    icon.style.display = 'block';
    texts.forEach(t => t.style.display = 'block');
    };

    uploadBox.addEventListener('dragover', e => {
    e.preventDefault();
    uploadBox.style.borderColor = '#333';
    uploadBox.style.backgroundColor = '#eee';
    });

    uploadBox.addEventListener('dragleave', e => {
    e.preventDefault();
    uploadBox.style.borderColor = '#999';
    uploadBox.style.backgroundColor = '#fff';
    });

    uploadBox.addEventListener('drop', e => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;
        showPreview(files[0]);
    }
    uploadBox.style.borderColor = '#999';
    });

    fileInput.addEventListener('change', e => {
    const file = e.target.files[0];
    showPreview(file);
    });

    deleteBtn.addEventListener('click', () => {
    resetUploadBox();
    });

    // 손 선택: 택1
    const handButtons = document.querySelectorAll('.setting-group:nth-of-type(1) .btn-group button');
    handButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        const isActive = btn.classList.contains('active');
        handButtons.forEach(b => b.classList.remove('active'));
        if (!isActive) btn.classList.add('active'); // 토글
        checkReadyToRecommend();
    });
    });


    // 용도 및 스타일
    const styleButtons = document.querySelectorAll('.setting-group:nth-of-type(2) .btn-group button');
    const autoBtn = [...styleButtons].find(b => b.textContent.trim() === '자동 추천');

    styleButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        const text = btn.textContent.trim();
        const isSelected = btn.classList.contains('active');

        if (text === '자동 추천') {
        styleButtons.forEach(b => {
            if (b !== btn) b.classList.remove('active');
        });
        btn.classList.toggle('active', !isSelected);
        } else {
        autoBtn.classList.remove('active');

        if (['맥시멀리스트', '미니멀리스트'].includes(text)) {
            const other = [...styleButtons].find(b =>
            ['맥시멀리스트', '미니멀리스트'].includes(b.textContent.trim()) &&
            b !== btn
            );
            if (isSelected) {
            btn.classList.remove('active');
            } else {
            other?.classList.remove('active');
            btn.classList.add('active');
            }
        } else {
            btn.classList.toggle('active', !isSelected);
        }
        }
        checkReadyToRecommend();
    });
    });


  //배치 추천 버튼 이벤트
    const recommendBtn = document.getElementById('recommendBtn');

    function checkReadyToRecommend() {
    const hasImage = fileInput.files.length > 0;
    const hasHand = [...handButtons].some(b => b.classList.contains('active'));
    const hasStyle = [...styleButtons].some(b => b.classList.contains('active'));

    if (hasImage && hasHand && hasStyle) {
        recommendBtn.disabled = false;
        recommendBtn.classList.add('enabled');
    } else {
        recommendBtn.disabled = true;
        recommendBtn.classList.remove('enabled');
    }
    }

    // 이벤트마다 체크
    fileInput.addEventListener('change', checkReadyToRecommend);
    deleteBtn.addEventListener('click', checkReadyToRecommend);

    handButtons.forEach(btn => {
    btn.addEventListener('click', checkReadyToRecommend);
    });

    styleButtons.forEach(btn => {
    btn.addEventListener('click', checkReadyToRecommend);
    });
});