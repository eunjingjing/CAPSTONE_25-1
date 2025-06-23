document.addEventListener('DOMContentLoaded', () => {
    const uploadBox = document.querySelector('.upload-box');
    const fileInput = document.getElementById('imgUpload');
    const icon = uploadBox.querySelector('.material-symbols-outlined');
    const texts = [...uploadBox.querySelectorAll('div, label')];
    const recommendBtn = document.getElementById('recommendBtn');

    const previewWrapper = document.createElement('div');
    previewWrapper.style.position = 'relative';
    previewWrapper.style.display = 'none';
    uploadBox.appendChild(previewWrapper);

    const previewImg = document.createElement('img');
    previewImg.style.maxWidth = '30vw';
    previewImg.style.borderRadius = '10px';
    previewImg.style.marginTop = '10px';
    previewWrapper.appendChild(previewImg);

    const deleteBtn = document.createElement('button');
    deleteBtn.textContent = '✕';
    Object.assign(deleteBtn.style, {
        position: 'absolute', top: '-10px', right: '-10px',
        background: '#f44336', color: '#fff', border: 'none',
        borderRadius: '50%', width: '25px', height: '25px',
        cursor: 'pointer', fontSize: '16px'
    });
    previewWrapper.appendChild(deleteBtn);

    const showPreview = file => {
        if (!file || !file.type.startsWith('image/')) return;
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
        checkReady();
    };

    fileInput.addEventListener('change', e => { showPreview(e.target.files[0]); checkReady(); });
    deleteBtn.addEventListener('click', resetUploadBox);
    uploadBox.addEventListener('dragover', e => { e.preventDefault(); uploadBox.style.borderColor = '#333'; uploadBox.style.backgroundColor = '#eee'; });
    uploadBox.addEventListener('dragleave', e => { e.preventDefault(); uploadBox.style.borderColor = '#999'; uploadBox.style.backgroundColor = '#fff'; });
    uploadBox.addEventListener('drop', e => {
        e.preventDefault();
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            showPreview(files[0]);
            checkReady();
        }
        uploadBox.style.borderColor = '#999';
    });

    const handButtons = document.querySelectorAll('.hand-group .btn-group button');
    const lifestyleButtons = document.querySelectorAll('.lifestyle-group .btn-group button');
    const purposeButtons = document.querySelectorAll('.purpose-group .btn-group button');
    const autoButton = document.querySelector('.auto-btn button');

    handButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const isActive = btn.classList.contains('active');
            handButtons.forEach(b => b.classList.remove('active'));
            if (!isActive) btn.classList.add('active');
            checkReady();
        });
    });

    lifestyleButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const isActive = btn.classList.contains('active');
            lifestyleButtons.forEach(b => b.classList.remove('active'));
            if (!isActive) btn.classList.add('active');
            autoButton.classList.remove('active');
            checkReady();
        });
    });

    purposeButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            btn.classList.toggle('active');
            autoButton.classList.remove('active');
            checkReady();
        });
    });

    autoButton.addEventListener('click', () => {
        const isActive = autoButton.classList.contains('active');
        if (!isActive) {
            lifestyleButtons.forEach(b => b.classList.remove('active'));
            purposeButtons.forEach(b => b.classList.remove('active'));
            autoButton.classList.add('active');
        } else {
            autoButton.classList.remove('active');
        }
        checkReady();
    });

    function checkReady() {
        const hasImage = fileInput.files.length > 0;
        const handSelected = [...handButtons].some(b => b.classList.contains('active'));
        const autoSelected = autoButton.classList.contains('active');
        const lifestyleSelected = [...lifestyleButtons].some(b => b.classList.contains('active'));
        const purposeSelectedCount = [...purposeButtons].filter(b => b.classList.contains('active')).length;

        let valid = false;
        if (autoSelected) {
            valid = hasImage && handSelected;
        } else {
            valid = hasImage && handSelected && lifestyleSelected && (purposeSelectedCount >= 1);
        }

        recommendBtn.disabled = !valid;
        recommendBtn.classList.toggle('enabled', valid);
    }

    fileInput.addEventListener('change', checkReady);

    document.getElementById("recommendBtn").addEventListener("click", (e) => {
        e.preventDefault();
        const hand = document.querySelector(".hand-group .btn-group .active")?.textContent.trim() || "";
        const lifestyle = document.querySelector(".lifestyle-group .btn-group .active")?.textContent.trim() || "";
        const purposeElems = document.querySelectorAll(".purpose-group .btn-group .active");
        const purposes = [...purposeElems].map(el => el.textContent.trim()).join(",");

        document.getElementById("handInput").value = hand;
        document.getElementById("lifestyleInput").value = lifestyle;
        document.getElementById("purposeInput").value = purposes;

        console.log("✅ recommendForm submit 직전");
        document.getElementById("recommendForm").submit();
    });
});