// animations.js
document.addEventListener('DOMContentLoaded', function() {
    // Loading animation for all pages
    const initLoadingAnimation = () => {
        const loadingBar = document.createElement('div');
        loadingBar.className = 'progress-bar mx-auto w-64 mb-4';
        loadingBar.innerHTML = '<div class="progress-fill" id="loadingBar"></div>';

        const mainTitle = document.querySelector('h1');
        if (mainTitle) {
            mainTitle.insertAdjacentElement('afterend', loadingBar);
            setTimeout(() => {
                document.getElementById('loadingBar').style.width = '100%';
                setTimeout(() => {
                    loadingBar.style.opacity = '0';
                    setTimeout(() => loadingBar.remove(), 300);
                }, 800);
            }, 50);
        }
    };

    // Animate cards/stats with fade-in effect
    const animateCards = () => {
        const cards = document.querySelectorAll('.stat-card, .feature-card, .cluster-card');
        cards.forEach((card, index) => {
            setTimeout(() => {
                card.classList.add('fade-in');
            }, index * 150);
        });
    };

    // Animate buttons with staggered effect
    const animateButtons = () => {
        const buttons = document.querySelectorAll('.domain-btn, .algorithm-btn');
        buttons.forEach((btn, index) => {
            setTimeout(() => {
                btn.classList.add('fade-in');
                btn.style.transform = 'scale(1)';
            }, 300 + (index * 50));
        });
    };

    // Table row hover effects
    const initTableHover = () => {
        const tableRows = document.querySelectorAll('.data-table tr');
        tableRows.forEach(row => {
            row.addEventListener('mouseenter', () => {
                row.classList.add('bg-gray-50');
            });
            row.addEventListener('mouseleave', () => {
                row.classList.remove('bg-gray-50');
            });
        });
    };

    // Pulse hover effect
    const initPulseHover = () => {
        const pulseElements = document.querySelectorAll('.pulse-hover');
        pulseElements.forEach(el => {
            el.addEventListener('mouseenter', () => {
                el.classList.add('animate-pulse');
            });
            el.addEventListener('mouseleave', () => {
                el.classList.remove('animate-pulse');
            });
        });
    };

    // Initialize all animations
    initLoadingAnimation();
    animateCards();
    animateButtons();
    initTableHover();
    initPulseHover();
});