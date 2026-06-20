/**
 * Main.js - Medical Assistant Frontend Interaction Logic
 * Modern UI interactions, keyboard shortcuts, and utility functions
 */

// ============================================================
// 1. Keyboard Shortcuts
// ============================================================
document.addEventListener('DOMContentLoaded', function() {
    // Ctrl+Shift+C: Clear chat
    document.addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.shiftKey && e.key === 'C') {
            e.preventDefault();
            const clearBtn = document.getElementById('clear-chat');
            if (clearBtn) clearBtn.click();
        }
    });

    // Ctrl+/: Focus message input
    document.addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.key === '/') {
            e.preventDefault();
            const input = document.getElementById('message-input');
            if (input) input.focus();
        }
    });

    // Escape: Close sidebar on mobile
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            const sidebar = document.getElementById('sidebar');
            if (sidebar && sidebar.classList.contains('open')) {
                sidebar.classList.remove('open');
            }
        }
    });
});

// ============================================================
// 2. Textarea Auto-resize
// ============================================================
document.addEventListener('DOMContentLoaded', function() {
    const textarea = document.getElementById('message-input');
    if (!textarea) return;

    textarea.addEventListener('input', function() {
        this.style.height = 'auto';
        const maxHeight = 150; // px
        this.style.height = Math.min(this.scrollHeight, maxHeight) + 'px';
    });

    // Reset height on form submit
    const form = document.getElementById('chat-form');
    if (form) {
        form.addEventListener('submit', function() {
            setTimeout(() => {
                textarea.style.height = 'auto';
            }, 100);
        });
    }
});

// ============================================================
// 3. Scroll to Bottom Button
// ============================================================
document.addEventListener('DOMContentLoaded', function() {
    const chatBody = document.getElementById('chat-body');
    if (!chatBody) return;

    // Create scroll-to-bottom button
    const scrollBtn = document.createElement('button');
    scrollBtn.id = 'scroll-bottom-btn';
    scrollBtn.innerHTML = '<i class="fas fa-chevron-down"></i>';
    scrollBtn.style.cssText = `
        position: absolute;
        bottom: 80px;
        right: 20px;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: var(--primary);
        color: white;
        border: none;
        cursor: pointer;
        display: none;
        align-items: center;
        justify-content: center;
        box-shadow: var(--shadow-md);
        z-index: 100;
        transition: opacity 0.2s ease, transform 0.2s ease;
        font-size: 14px;
    `;
    scrollBtn.addEventListener('click', function() {
        chatBody.scrollTo({
            top: chatBody.scrollHeight,
            behavior: 'smooth'
        });
    });
    chatBody.parentElement.style.position = 'relative';
    chatBody.parentElement.appendChild(scrollBtn);

    // Show/hide based on scroll position
    chatBody.addEventListener('scroll', function() {
        const threshold = 200;
        const isNearBottom = chatBody.scrollHeight - chatBody.scrollTop - chatBody.clientHeight < threshold;
        scrollBtn.style.display = isNearBottom ? 'none' : 'flex';
    });
});

// ============================================================
// 4. Markdown Rendering (marked.js integration)
// ============================================================
document.addEventListener('DOMContentLoaded', function() {
    if (typeof marked !== 'undefined') {
        marked.setOptions({
            breaks: true,
            gfm: true,
            headerIds: false,
            mangle: false
        });
    }
});

// ============================================================
// 5. Image Preview with Lightbox
// ============================================================
document.addEventListener('DOMContentLoaded', function() {
    // Delegate click on message images
    document.addEventListener('click', function(e) {
        const img = e.target.closest('.message-content img, .image-container img');
        if (!img) return;

        // Create lightbox overlay
        const overlay = document.createElement('div');
        overlay.style.cssText = `
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
            cursor: pointer;
            animation: fadeIn 0.2s ease;
        `;

        const fullImg = document.createElement('img');
        fullImg.src = img.src;
        fullImg.style.cssText = `
            max-width: 90vw;
            max-height: 90vh;
            border-radius: 12px;
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5);
        `;

        overlay.appendChild(fullImg);
        document.body.appendChild(overlay);

        // Close on click or Escape
        overlay.addEventListener('click', () => overlay.remove());
        const closeHandler = (e) => {
            if (e.key === 'Escape') {
                overlay.remove();
                document.removeEventListener('keydown', closeHandler);
            }
        };
        document.addEventListener('keydown', closeHandler);
    });
});

// ============================================================
// 6. File Upload Drag & Drop
// ============================================================
document.addEventListener('DOMContentLoaded', function() {
    const chatBody = document.getElementById('chat-body');
    const chatFooter = document.querySelector('.chat-footer');
    const target = chatFooter || chatBody;
    if (!target) return;

    ['dragenter', 'dragover'].forEach(event => {
        target.addEventListener(event, function(e) {
            e.preventDefault();
            e.stopPropagation();
            this.style.borderColor = 'var(--primary)';
            this.style.background = 'var(--primary-light)';
        });
    });

    ['dragleave', 'drop'].forEach(event => {
        target.addEventListener(event, function(e) {
            e.preventDefault();
            e.stopPropagation();
            this.style.borderColor = '';
            this.style.background = '';
        });
    });

    target.addEventListener('drop', function(e) {
        const files = e.dataTransfer.files;
        if (files.length === 0) return;

        const file = files[0];
        if (!file.type.startsWith('image/')) {
            showNotification('Please drop an image file', 'warning');
            return;
        }

        // Trigger image upload via existing input
        const imageUpload = document.getElementById('image-upload');
        if (imageUpload) {
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            imageUpload.files = dataTransfer.files;
            imageUpload.dispatchEvent(new Event('change'));
        }
    });
});

// ============================================================
// 7. Notification System
// ============================================================
function showNotification(message, type = 'info', duration = 3000) {
    const notification = document.createElement('div');
    const colors = {
        info: 'var(--primary)',
        success: 'var(--success)',
        warning: 'var(--warning)',
        error: 'var(--danger)'
    };
    const icons = {
        info: 'fa-info-circle',
        success: 'fa-check-circle',
        warning: 'fa-exclamation-triangle',
        error: 'fa-times-circle'
    };

    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 20px;
        background: ${colors[type]};
        color: white;
        border-radius: 12px;
        font-family: var(--font-family);
        font-size: 14px;
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        z-index: 10001;
        display: flex;
        align-items: center;
        gap: 8px;
        animation: slideInRight 0.3s ease;
        max-width: 360px;
    `;
    notification.innerHTML = `<i class="fas ${icons[type]}"></i> ${message}`;
    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateX(100%)';
        notification.style.transition = 'all 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, duration);
}

// ============================================================
// 8. Session Timer
// ============================================================
document.addEventListener('DOMContentLoaded', function() {
    const sessionTimer = document.getElementById('session-timer');
    if (!sessionTimer) return;

    let seconds = 0;
    setInterval(() => {
        seconds++;
        const mins = Math.floor(seconds / 60).toString().padStart(2, '0');
        const secs = (seconds % 60).toString().padStart(2, '0');
        sessionTimer.textContent = `${mins}:${secs}`;
    }, 1000);
});

// ============================================================
// 9. Theme Toggle (for future dark mode)
// ============================================================
function toggleTheme() {
    document.body.classList.toggle('dark-theme');
    const isDark = document.body.classList.contains('dark-theme');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
}

// Restore theme on load
document.addEventListener('DOMContentLoaded', function() {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-theme');
    }
});
