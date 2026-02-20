/**
 * Custom User Interaction Trackers
 * 
 * This script handles custom tracking for:
 * 1. Page Load Performance
 * 2. Scroll Depth
 * 3. Click Tracking
 * 
 * Note: Currently logs to console. In a production environment, 
 * these would be sent to a backend analytics endpoint.
 */

(function() {
    'use strict';

    const debug = false;

    function logEvent(category, action, label = null, value = null) {
        if (debug) {
            console.log(`[Tracker] ${category}: ${action}`, { label, value });
        }
        
        // Integration point for Google Analytics or other backends
        if (typeof window.gtag === 'function') {
            window.gtag('event', action, {
                'event_category': category,
                'event_label': label,
                'value': value
            });
        }
    }

    // 1. Page Load Performance
    window.addEventListener('load', () => {
        // Wait a brief moment for performance API to settle
        setTimeout(() => {
            if (window.performance) {
                const navEntry = performance.getEntriesByType('navigation')[0];
                if (navEntry) {
                    const loadTime = Math.round(navEntry.loadEventEnd - navEntry.startTime);
                    logEvent('Performance', 'Page Load', 'Duration (ms)', loadTime);
                }
            }
        }, 0);
    });

    // 2. Scroll Depth Tracking
    let maxScroll = 0;
    const scrollMilestones = [25, 50, 75, 90];
    const reachedMilestones = new Set();

    function trackScroll() {
        const scrollTop = window.scrollY || document.documentElement.scrollTop;
        const docHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
        const scrollPercent = Math.round((scrollTop / docHeight) * 100);

        if (scrollPercent > maxScroll) {
            maxScroll = scrollPercent;
        }

        scrollMilestones.forEach(milestone => {
            if (scrollPercent >= milestone && !reachedMilestones.has(milestone)) {
                reachedMilestones.add(milestone);
                logEvent('Engagement', 'Scroll Depth', `${milestone}%`);
            }
        });
    }

    // Throttle scroll event
    let scrollTimeout;
    window.addEventListener('scroll', () => {
        if (!scrollTimeout) {
            scrollTimeout = setTimeout(() => {
                trackScroll();
                scrollTimeout = null;
            }, 100);
        }
    });

    // 3. Click Tracking
    document.addEventListener('click', (event) => {
        const link = event.target.closest('a');
        if (link) {
            const href = link.getAttribute('href');
            const text = link.innerText.trim();
            
            // Distinguish between internal and external links
            const isExternal = href && (href.startsWith('http') || href.startsWith('//')) && !href.includes(window.location.hostname);
            
            if (isExternal) {
                logEvent('Interaction', 'External Click', href);
            } else {
                logEvent('Interaction', 'Internal Click', href);
            }
        }
    });

    if (debug) console.log('[Tracker] Initialized');

})();
