(function() {
    'use strict';
    var state = { isOpen: false, messages: [], isLoading: false };
    var currentCitations = null;
    var hasOpenedChat = false;
    var userHasScrolled = false;
    var STORAGE_KEY_VISITOR = 'geoai-book-visitor-id';
    var STORAGE_KEY_CONVERSATION = 'geoai-book-conversation-id';
    var STORAGE_KEY_LAST_MSG = 'geoai-book-last-message-ts';
    var INACTIVITY_MS = 30 * 60 * 1000;
    var visitorId = null;
    var conversationId = null;
    var currentInteractionId = null;
    var contextWindow = 0;
    var turnCount = 0;
    var isExpanded = false;
    try { visitorId = localStorage.getItem(STORAGE_KEY_VISITOR); } catch (_e) {}
    try { conversationId = localStorage.getItem(STORAGE_KEY_CONVERSATION); } catch (_e) {}
    try { isExpanded = localStorage.getItem('geoai-chat-expanded') === 'true'; } catch (_e) {}
    var SUGGESTED_QUESTIONS = [
        "What is semantic segmentation in the context of remote sensing?",
        "How does the book use deep learning to predict crop yield?",
        "Why use an Earth observation foundation model instead of training my own?",
        "How is ethics applied in GeoAI?"
    ];
    var panel = document.getElementById('geoai-chat-panel');
    var toggle = document.getElementById('geoai-chat-toggle');
    var closeBtn = document.getElementById('geoai-chat-close');
    var newChatBtn = document.getElementById('geoai-chat-new');
    var suggestions = document.getElementById('geoai-chat-suggestions');
    var messagesEl = document.getElementById('geoai-chat-messages');
    var statusEl = document.getElementById('geoai-chat-status');
    var form = document.getElementById('geoai-chat-form');
    var textarea = form.querySelector('textarea');
    var sendBtn = form.querySelector('button[type="submit"]');
    var expandBtn = document.getElementById('geoai-chat-expand');
    var contextEl = document.getElementById('geoai-chat-context');
    var contextWarnEl = document.getElementById('geoai-chat-context-warn');
    var contextFill = contextEl.querySelector('.geoai-ctx-fill');
    var contextNums = contextEl.querySelector('.geoai-ctx-nums');
    var contextPct = contextEl.querySelector('.geoai-ctx-pct');

    if (isExpanded) {
        expandBtn.innerHTML = '&#x2923;';
        expandBtn.setAttribute('aria-label', 'Collapse to panel');
    }

    SUGGESTED_QUESTIONS.forEach(function(q) {
        var btn = document.createElement('button');
        btn.type = 'button';
        btn.textContent = q;
        btn.addEventListener('click', function() { sendMessage(q); });
        suggestions.appendChild(btn);
    });

    toggle.addEventListener('click', toggleChat);

    closeBtn.addEventListener('click', toggleChat);

    newChatBtn.addEventListener('click', function() {
        if (state.isLoading) return;
        conversationId = null;
        try { localStorage.removeItem(STORAGE_KEY_CONVERSATION); } catch (_e) {}
        try { localStorage.removeItem(STORAGE_KEY_LAST_MSG); } catch (_e) {}
        state.messages = [];
        currentCitations = null;
        contextWindow = 0;
        turnCount = 0;
        contextEl.hidden = true;
        contextWarnEl.hidden = true;
        panel.classList.remove('ctx-full');
        textarea.disabled = false;
        textarea.placeholder = 'Ask about the book...';
        sendBtn.disabled = false;
        messagesEl.innerHTML = '';
        suggestions.classList.remove('hidden');
        statusEl.textContent = '';
        textarea.focus();
    });

    expandBtn.addEventListener('click', toggleExpanded);

    textarea.addEventListener('input', function() {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    });

    textarea.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            form.dispatchEvent(new Event('submit', { cancelable: true }));
        }
    });

    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && state.isOpen) toggleChat();
    });

    panel.addEventListener('keydown', function(e) {
        if (e.key !== 'Tab') return;
        var focusable = Array.from(panel.querySelectorAll(
            'button:not([disabled]), textarea:not([disabled]), a[href], [tabindex]:not([tabindex="-1"])'
        )).filter(function(el) { return el.offsetParent !== null; });
        if (!focusable.length) return;
        var first = focusable[0], last = focusable[focusable.length - 1];
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    });

    messagesEl.addEventListener('scroll', function() {
        userHasScrolled = messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight > 40;
    });

    function toggleChat() {
        state.isOpen = !state.isOpen;
        panel.classList.toggle('open', state.isOpen);
        panel.inert = !state.isOpen;
        panel.setAttribute('aria-hidden', String(!state.isOpen));
        toggle.setAttribute('aria-label', state.isOpen ? 'Close chat assistant' : 'Open chat assistant');
        toggle.setAttribute('aria-expanded', String(state.isOpen));
        if (state.isOpen) {
            if (isExpanded) {
                panel.classList.add('expanded');
                document.body.classList.add('geoai-sidebar-open');
                toggle.style.display = 'none';
            }
            textarea.focus();
            if (!hasOpenedChat) {
                hasOpenedChat = true;
                if (typeof gtag === 'function') gtag('event', 'chat_open', { event_category: 'chat' });
            }
        } else {
            document.body.classList.remove('geoai-sidebar-open');
            toggle.style.display = '';
        }
    }

    /**
     * Toggle between floating panel and full-height sidebar mode.
     */
    function toggleExpanded() {
        isExpanded = !isExpanded;
        panel.classList.toggle('expanded', isExpanded);
        document.body.classList.toggle('geoai-sidebar-open', isExpanded);
        expandBtn.innerHTML = isExpanded ? '&#x2923;' : '&#x2922;';
        expandBtn.setAttribute('aria-label', isExpanded ? 'Collapse to panel' : 'Expand to sidebar');
        try { localStorage.setItem('geoai-chat-expanded', String(isExpanded)); } catch (_e) {}
        toggle.style.display = isExpanded ? 'none' : '';
    }

    function autoScroll() {
        if (!userHasScrolled) messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function escapeHtml(text) {
        return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }

    function renderMarkdown(text) {
        var html = escapeHtml(text);
        var codeBlocks = [];
        // 1. Fenced code blocks -> placeholder
        html = html.replace(/```([\s\S]*?)```/g, function(_, code) {
            codeBlocks.push('<pre><code>' + code + '</code></pre>');
            return '\x00CB' + (codeBlocks.length - 1) + '\x00';
        });
        // 2. Inline code -> placeholder (protects content from link/bold/italic)
        html = html.replace(/`([^`]+)`/g, function(_, code) {
            codeBlocks.push('<code>' + code + '</code>');
            return '\x00CB' + (codeBlocks.length - 1) + '\x00';
        });
        // 3. Links -> placeholder (https only; protects href from bold/italic)
        html = html.replace(/\[([^\]]+)\]\((https:\/\/[^\s)]+)\)/g, function(match, linkText, url) {
            if (/&quot;|&#39;/.test(url)) return match;
            var tag = '<a href="' + url + '" target="_blank" rel="noopener noreferrer" class="md-link">' + linkText + '</a>';
            codeBlocks.push(tag);
            return '\x00CB' + (codeBlocks.length - 1) + '\x00';
        });
        // 4. Heading normalization (ensure blank lines around headings)
        html = html.replace(/([^\n])\n(#{2,4}\s)/gm, '$1\n\n$2');
        html = html.replace(/(^#{2,4}\s.+$)\n([^\n])/gm, '$1\n\n$2');
        // 5. Heading conversion (## -> h3, ### -> h4, #### -> h5)
        html = html.replace(/^(#{2,4})\s+(.+)$/gm, function(_, hashes, headingText) {
            var level = hashes.length + 1;
            return '<h' + level + '>' + headingText.trim() + '</h' + level + '>';
        });
        // 6. Bold / italic
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/(^|[^*])\*([^*]+)\*(?!\*)/g, '$1<em>$2</em>');
        // 7a. Collapse blank lines between list items so the block regex sees one
        // contiguous list. Handles multi-line items: a marker line may be followed by
        // indented continuation lines before the blank separator. The loop handles
        // chains (collapsing 1-2 may make 2-3 adjacent).
        var prevHtml;
        do {
            prevHtml = html;
            html = html.replace(
                /(\n[ \t]*(?:\d+\.|-)[ \t][^\n]+(?:\n(?![ \t]*(?:\d+\.|-)[ \t])(?!\n)[^\n]+)*)\n\n([ \t]*(?:\d+\.|-)[ \t])/g,
                '$1\n$2'
            );
        } while (html !== prevHtml);
        // 7a-fold. Join continuation lines onto the preceding list item so each
        // logical item is one physical line for the block regex. Only folds lines
        // that are not themselves list markers and not blank.
        var foldRe = /(\n[ \t]*(?:\d+\.|-)[ \t]+[^\n]+)\n(?![ \t]*(?:\d+\.|-)[ \t])(?!\n)([^\n]+)/g;
        do {
            prevHtml = html;
            html = html.replace(foldRe, '$1 $2');
        } while (html !== prevHtml);
        // 7b. Lists (unified: supports nested - under \d+., tolerates indentation)
        html = html.replace(/((?:^|\n)[ \t]*(?:\d+\. |- ).+(?:\n[ \t]*(?:\d+\. |- ).+)*)/g, function(block) {
            var lines = block.trim().split('\n');
            var hasNumbered = lines.some(function(l) { return /^\s*\d+\. /.test(l); });
            var hasBullets = lines.some(function(l) { return /^\s*- /.test(l); });
            if (hasNumbered && hasBullets) {
                var result = '<ol>', inSub = false;
                for (var i = 0; i < lines.length; i++) {
                    var trimmed = lines[i].replace(/^\s+/, '');
                    if (/^\d+\. /.test(trimmed)) {
                        if (inSub) { result += '</ul>'; inSub = false; }
                        if (i > 0) result += '</li>';
                        result += '<li>' + trimmed.replace(/^\d+\. /, '');
                    } else {
                        if (!inSub) { result += '<ul>'; inSub = true; }
                        result += '<li>' + trimmed.replace(/^- /, '') + '</li>';
                    }
                }
                if (inSub) result += '</ul>';
                return result + '</li></ol>';
            }
            if (hasNumbered) {
                var items = lines.map(function(l) {
                    return '<li>' + l.replace(/^\s*\d+\. /, '') + '</li>';
                }).join('');
                return '<ol>' + items + '</ol>';
            }
            var items = lines.map(function(l) {
                return '<li>' + l.replace(/^\s*- /, '') + '</li>';
            }).join('');
            return '<ul>' + items + '</ul>';
        });
        // 8. Tables
        html = html.replace(/^(\|.+\|)[ \t]*\n(\|[\s:|-]+\|)[ \t]*\n((?:\|.+\|[ \t]*(?:\n|$))+)/gm, function(_, headerLine, _sep, bodyBlock) {
            var headers = headerLine.split('|').slice(1, -1);
            var rows = bodyBlock.trim().split('\n');
            var thead = '<thead><tr>' + headers.map(function(h) {
                return '<th>' + h.trim() + '</th>';
            }).join('') + '</tr></thead>';
            var tbody = '<tbody>' + rows.map(function(row) {
                var cells = row.split('|').slice(1, -1);
                return '<tr>' + cells.map(function(c) {
                    return '<td>' + c.trim() + '</td>';
                }).join('') + '</tr>';
            }).join('') + '</tbody>';
            return '<div class="table-wrap"><table>' + thead + tbody + '</table></div>';
        });
        // 9. Horizontal rules
        html = html.replace(/^-{3,}$/gm, '<hr>');
        // 10. Paragraphs
        html = html.replace(/\n\n+/g, '</p><p>');
        html = '<p>' + html + '</p>';
        html = html.replace(/<p>\s*<\/p>/g, '');
        // 11. Restore placeholders (before block cleanup so <pre>/<h3> are real tags)
        html = html.replace(/\x00CB(\d+)\x00/g, function(_, i) {
            return codeBlocks[parseInt(i, 10)];
        });
        // 12. Block element cleanup (unwrap from <p> nesting)
        html = html.replace(/<p>\s*(<(?:ul|ol|pre|h[3-5]|hr|div))/g, '$1');
        html = html.replace(/(<\/(?:ul|ol|pre|h[3-5]|div)>)\s*<\/p>/g, '$1');
        html = html.replace(/(<hr>)\s*<\/p>/g, '$1');
        return html;
    }

    function linkCitations(messageEl) {
        if (!currentCitations || !currentCitations.length) return;
        var walker = document.createTreeWalker(messageEl, NodeFilter.SHOW_TEXT, {
            acceptNode: function(node) {
                var parent = node.parentElement;
                if (!parent) return NodeFilter.FILTER_REJECT;
                if (parent.closest('code, pre, a, .sources')) return NodeFilter.FILTER_REJECT;
                return /\[\d+\]/.test(node.nodeValue) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP;
            }
        });
        var textNodes = [];
        while (walker.nextNode()) textNodes.push(walker.currentNode);
        textNodes.forEach(function(node) {
            var pattern = /\[(\d+)\]/g, parts = [], lastIndex = 0, match;
            while ((match = pattern.exec(node.nodeValue)) !== null) {
                var num = parseInt(match[1], 10);
                if (num < 1 || num > currentCitations.length) continue;
                var citation = currentCitations[num - 1];
                if (!citation || typeof citation !== 'object') continue;
                if (match.index > 0 && /\w/.test(node.nodeValue[match.index - 1])) continue;
                if (match.index > lastIndex)
                    parts.push(document.createTextNode(node.nodeValue.slice(lastIndex, match.index)));
                if (citation.url) {
                    var a = document.createElement('a');
                    a.href = citation.url;
                    a.textContent = '[' + num + ']';
                    a.title = citation.display_label || (citation.chapter ? citation.chapter + (citation.section ? ' - ' + citation.section : '') : citation.section || '');
                    a.target = '_blank'; a.rel = 'noopener noreferrer'; a.className = 'citation-ref';
                    a.addEventListener('click', (function(c) {
                        return function() {
                            if (typeof gtag === 'function') gtag('event', 'citation_clicked', {
                                event_category: 'chat', citation_chapter: c.chapter || '', citation_section: c.section || ''
                            });
                        };
                    })(citation));
                    parts.push(a);
                } else {
                    var span = document.createElement('span');
                    span.textContent = '[' + num + ']';
                    span.className = 'citation-ref-plain';
                    parts.push(span);
                }
                lastIndex = match.index + match[0].length;
            }
            if (!parts.length) return;
            if (lastIndex < node.nodeValue.length)
                parts.push(document.createTextNode(node.nodeValue.slice(lastIndex)));
            var frag = document.createDocumentFragment();
            parts.forEach(function(p) { frag.appendChild(p); });
            node.parentNode.replaceChild(frag, node);
        });
    }

    function renderSources(messageEl, citations) {
        if (!citations || !citations.length) return;
        var groups = Object.create(null);
        var groupOrder = [];
        citations.forEach(function(c, i) {
            if (!c || typeof c !== 'object') return;
            var key = (c.source_path || c.url || 'idx-' + i) + '\x00' + (c.display_label || '');
            if (!groups[key]) {
                groups[key] = { entries: [], citation: c };
                groupOrder.push(key);
            }
            groups[key].entries.push({ index: i + 1, url: c.url });
        });
        if (!groupOrder.length) return;
        var details = document.createElement('details');
        details.className = 'sources';
        var summary = document.createElement('summary');
        summary.textContent = 'Sources';
        details.appendChild(summary);
        var ul = document.createElement('ul');
        groupOrder.forEach(function(key) {
            var group = groups[key];
            var c = group.citation;
            var li = document.createElement('li');
            group.entries.forEach(function(entry) {
                var ref = '[' + entry.index + ']';
                if (entry.url) {
                    var a = document.createElement('a');
                    a.href = entry.url; a.textContent = ref;
                    a.target = '_blank'; a.rel = 'noopener noreferrer';
                    a.className = 'source-ref';
                    li.appendChild(a);
                } else {
                    var span = document.createElement('span');
                    span.textContent = ref;
                    span.className = 'source-ref-plain';
                    li.appendChild(span);
                }
            });
            var text = c.display_label
                || (c.chapter ? c.chapter + (c.section ? ' - ' + c.section : '') : c.section || 'Source');
            var label = document.createTextNode(' ' + text);
            li.appendChild(label);
            ul.appendChild(li);
        });
        details.appendChild(ul);
        messageEl.appendChild(details);
    }

    function createBubble(type) {
        var div = document.createElement('div');
        div.className = 'msg msg-' + type;
        messagesEl.appendChild(div);
        autoScroll();
        return div;
    }

    function showLoading() {
        var bubble = createBubble('assistant');
        bubble.innerHTML = '<div class="loading-dots"><span></span><span></span><span></span></div>';
        bubble.dataset.loading = 'true';
        return bubble;
    }

    function clearLoading(bubble) {
        if (bubble && bubble.dataset.loading) { bubble.innerHTML = ''; delete bubble.dataset.loading; }
    }

    function setControls(disabled) { textarea.disabled = disabled; sendBtn.disabled = disabled; }

    function restoreSuggestions() {
        if (state.messages.length === 0) suggestions.classList.remove('hidden');
    }

    function findDataLine(eventText) {
        var lines = eventText.split('\n');
        for (var i = 0; i < lines.length; i++)
            if (lines[i].indexOf('data:') === 0) return lines[i];
        return null;
    }

    function getGa4ClientId() {
        try {
            var match = document.cookie.match(/(?:^|;\s*)_ga=GA\d+\.\d+\.(.+?)(?:;|$)/);
            return match ? match[1] : null;
        } catch (_e) { return null; }
    }

    /**
     * Update the context usage bar with current token counts.
     * @param {number} used - Prompt tokens consumed.
     * @param {number} total - Total context window size in tokens.
     */
    function updateContextBar(used, total) {
        turnCount += 1;
        var pct = Math.round(used / total * 100);
        var usedStr = used >= 1000 ? (used / 1000).toFixed(1) + 'k' : String(used);
        var totalStr = total >= 1000 ? (total / 1000).toFixed(1) + 'k' : String(total);
        var level = pct < 80 ? 'normal' : pct < 95 ? 'warn' : 'critical';
        contextFill.style.width = pct + '%';
        contextFill.className = 'geoai-ctx-fill ' + level;
        contextNums.textContent = usedStr + ' / ' + totalStr + '  ·  Turn ' + turnCount;
        contextPct.textContent = pct + '%';
        contextPct.className = 'geoai-ctx-pct ' + level;
        contextEl.hidden = false;
        if (pct >= 100) {
            contextWarnEl.textContent = 'Context full. Please start a new chat to continue.';
            contextWarnEl.hidden = false;
            textarea.disabled = true;
            textarea.placeholder = 'Context full — start a new chat';
            sendBtn.disabled = true;
            panel.classList.add('ctx-full');
        } else if (pct >= 95) {
            contextWarnEl.textContent = 'Context nearly full. Start a new chat for best results.';
            contextWarnEl.hidden = false;
            textarea.disabled = false;
            textarea.placeholder = 'Ask about the book...';
            sendBtn.disabled = false;
            panel.classList.remove('ctx-full');
        } else {
            contextWarnEl.hidden = true;
            textarea.disabled = false;
            textarea.placeholder = 'Ask about the book...';
            sendBtn.disabled = false;
            panel.classList.remove('ctx-full');
        }
    }

    /**
     * Derive a short display name from a citation's source path.
     * @param {Object} citation - Citation with source_path, chapter, section, display_label.
     * @returns {string} Human-readable short name for the source.
     */
    function humanizeSource(citation) {
        var path = citation.source_path || '';
        var bookMatch = path.match(/book\/0*(\d+)_([^\/]+)/);
        if (bookMatch) {
            var chNum = parseInt(bookMatch[1], 10);
            var secMatch = path.match(/\/\d+__?([^\/]+)$/);
            if (secMatch) {
                return 'Ch.' + chNum + ' - ' + secMatch[1].replace(/_/g, ' ');
            }
            return 'Ch.' + chNum + ' - ' + bookMatch[2].replace(/_/g, ' ');
        }
        if (path.indexOf('transcript') !== -1) return 'Video transcript';
        if (citation.chapter) {
            return citation.chapter + (citation.section ? ' - ' + citation.section : '');
        }
        if (citation.display_label) return citation.display_label;
        var seg = path.split('/').pop() || '';
        return seg.replace(/_/g, ' ') || 'Source';
    }

    /**
     * Render compact source chips below an assistant message bubble.
     * @param {HTMLElement} bubble - The assistant message bubble element.
     */
    function renderSourceChips(bubble) {
        if (!currentCitations || !currentCitations.length) return;
        var seen = Object.create(null);
        var unique = [];
        for (var i = 0; i < currentCitations.length; i++) {
            var c = currentCitations[i];
            if (!c || typeof c !== 'object') continue;
            var key = c.source_path || ('idx-' + i);
            if (!seen[key]) {
                seen[key] = true;
                unique.push(c);
            }
        }
        if (!unique.length) return;
        var container = document.createElement('div');
        container.className = 'geoai-source-chips';
        var limit = Math.min(unique.length, 4);
        for (var j = 0; j < limit; j++) {
            var chip = document.createElement('span');
            chip.className = 'geoai-source-chip';
            chip.textContent = humanizeSource(unique[j]);
            container.appendChild(chip);
        }
        if (unique.length > 4) {
            var more = document.createElement('span');
            more.className = 'geoai-source-chip';
            more.textContent = '+' + (unique.length - 4) + ' more';
            container.appendChild(more);
        }
        bubble.appendChild(container);
    }

    function commitDone(question, assistantText, bubble) {
        state.messages.push({ role: 'user', content: question });
        state.messages.push({ role: 'assistant', content: assistantText });
        bubble.innerHTML = renderMarkdown(assistantText);
        linkCitations(bubble);
        renderSourceChips(bubble);
        renderSources(bubble, currentCitations);
        renderActions(bubble, assistantText, currentCitations, currentInteractionId);
        statusEl.textContent = 'Assistant response complete.';
        state.isLoading = false;
        setControls(false);
        autoScroll();
    }

    function renderActions(bubble, rawText, citations, interactionId) {
        var container = document.createElement('div');
        container.className = 'gai-feedback';
        var copyText = rawText;
        if (citations && citations.length) {
            copyText += '\n\nSources\n' + formatSourcesText(citations);
        }
        renderCopyButton(container, copyText);
        if (interactionId) renderFeedbackButtons(container, interactionId, bubble);
        bubble.appendChild(container);
    }

    function formatSourcesText(citations) {
        var seen = Object.create(null);
        var lines = [];
        citations.forEach(function(c, i) {
            if (!c || typeof c !== 'object') return;
            var label = c.display_label
                || (c.chapter ? c.chapter + (c.section ? ' - ' + c.section : '') : c.section || 'Source');
            var key = (c.source_path || '') + '\x00' + label;
            if (seen[key]) {
                seen[key].refs.push(i + 1);
            } else {
                seen[key] = { label: label, refs: [i + 1], url: c.url || '' };
                lines.push(seen[key]);
            }
        });
        return lines.map(function(g) {
            var refs = g.refs.map(function(n) { return '[' + n + ']'; }).join('');
            var line = refs + ' ' + g.label;
            if (g.url) line += ' (' + g.url + ')';
            return line;
        }).join('\n');
    }

    function renderCopyButton(container, copyText) {
        var copyIcon = '<svg viewBox="0 0 24 24" width="14" height="14"><path d="M16 1H4a2 2 0 0 0-2 2v14h2V3h12V1zm3 4H8a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2zm0 16H8V7h11v14z"/></svg>';
        var checkIcon = '<svg viewBox="0 0 24 24" width="14" height="14"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/></svg>';
        var btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'gai-copy-btn';
        btn.setAttribute('aria-label', 'Copy');
        btn.innerHTML = copyIcon;
        btn.addEventListener('click', function() {
            try {
                navigator.clipboard.writeText(copyText).then(function() {
                    btn.innerHTML = checkIcon;
                    btn.classList.add('copied');
                    setTimeout(function() { btn.innerHTML = copyIcon; btn.classList.remove('copied'); }, 1500);
                });
            } catch (_e) {}
        });
        container.appendChild(btn);
    }

    function renderFeedbackButtons(container, interactionId, bubble) {
        var thumbsUp = document.createElement('button');
        thumbsUp.type = 'button';
        thumbsUp.className = 'gai-feedback-btn';
        thumbsUp.setAttribute('aria-label', 'Helpful');
        thumbsUp.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16"><path d="M2 20h2V8H2v12zm20-11a2 2 0 0 0-2-2h-6.31l.95-4.57.03-.32a1.5 1.5 0 0 0-.44-1.06L13.17 0 6.59 6.59A2 2 0 0 0 6 8v10a2 2 0 0 0 2 2h9a2 2 0 0 0 1.84-1.22l3.02-7.05A2 2 0 0 0 22 11V9z"/></svg>';
        var thumbsDown = document.createElement('button');
        thumbsDown.type = 'button';
        thumbsDown.className = 'gai-feedback-btn';
        thumbsDown.setAttribute('aria-label', 'Not helpful');
        thumbsDown.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16"><path d="M22 4h-2v12h2V4zM2 15a2 2 0 0 0 2 2h6.31l-.95 4.57-.03.32a1.5 1.5 0 0 0 .44 1.06L10.83 24l6.58-6.59A2 2 0 0 0 18 16V6a2 2 0 0 0-2-2H7a2 2 0 0 0-1.84 1.22l-3.02 7.05A2 2 0 0 0 2 13v2z"/></svg>';
        container.appendChild(thumbsUp);
        container.appendChild(thumbsDown);

        var commentBox = null;
        var selectedRating = null;

        function submitFeedback(rating, comment) {
            var payload = { interaction_id: interactionId, rating: rating };
            if (comment) payload.comment = comment;
            fetch('/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            }).catch(function() {});
        }

        function selectRating(rating) {
            selectedRating = rating;
            thumbsUp.classList.toggle('active', rating === 'up');
            thumbsDown.classList.toggle('active', rating === 'down');
            if (rating === 'down' && !commentBox) {
                commentBox = document.createElement('div');
                commentBox.className = 'gai-feedback-comment';
                var input = document.createElement('textarea');
                input.placeholder = 'What was wrong? (optional)';
                input.rows = 2;
                input.maxLength = 2000;
                var sendBtn = document.createElement('button');
                sendBtn.type = 'button';
                sendBtn.textContent = 'Send';
                sendBtn.addEventListener('click', function() {
                    var text = input.value.trim();
                    submitFeedback('down', text || null);
                    commentBox.remove();
                    commentBox = null;
                    thumbsDown.disabled = false;
                    autoScroll();
                });
                commentBox.appendChild(input);
                commentBox.appendChild(sendBtn);
                container.after(commentBox);
                autoScroll();
            } else if (rating === 'up') {
                if (commentBox) { commentBox.remove(); commentBox = null; }
                submitFeedback('up', null);
            }
        }

        thumbsUp.addEventListener('click', function() { selectRating('up'); });
        thumbsDown.addEventListener('click', function() {
            if (selectedRating === 'down' && commentBox) return;
            selectRating('down');
        });
    }

    async function sendMessage(question) {
        if (state.isLoading) return;
        question = (question || '').trim();
        if (!question) return;
        try {
            var lastTs = parseInt(localStorage.getItem(STORAGE_KEY_LAST_MSG), 10);
            if (lastTs && Date.now() - lastTs > INACTIVITY_MS && conversationId) {
                conversationId = null;
                localStorage.removeItem(STORAGE_KEY_CONVERSATION);
                state.messages = [];
                currentCitations = null;
                contextWindow = 0;
                turnCount = 0;
                contextEl.hidden = true;
                contextWarnEl.hidden = true;
                messagesEl.innerHTML = '';
            }
        } catch (_e) {}
        try { localStorage.setItem(STORAGE_KEY_LAST_MSG, String(Date.now())); } catch (_e) {}
        state.isLoading = true;
        setControls(true);
        suggestions.classList.add('hidden');
        currentCitations = null;
        currentInteractionId = null;
        userHasScrolled = false;
        var userBubble = createBubble('user');
        userBubble.textContent = question;
        renderCopyButton(userBubble, question);
        textarea.value = '';
        textarea.style.height = 'auto';
        var history = state.messages.slice(-10);
        var loadingBubble = showLoading();
        if (typeof gtag === 'function')
            gtag('event', 'question_asked', { event_category: 'chat', question_length: question.length });
        var assistantText = '';
        var receivedDone = false;
        try {
            var requestBody = { question: question, history: history };
            if (visitorId) requestBody.visitor_id = visitorId;
            if (conversationId) requestBody.conversation_id = conversationId;
            var ga4ClientId = getGa4ClientId();
            if (ga4ClientId) requestBody.ga4_client_id = ga4ClientId;
            var response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
            });
            if (!response.ok) {
                clearLoading(loadingBubble);
                loadingBubble.className = 'msg msg-error';
                loadingBubble.textContent = 'The assistant is temporarily unavailable.';
                state.isLoading = false; setControls(false); restoreSuggestions(); return;
            }
            var reader = response.body.getReader();
            var decoder = new TextDecoder();
            var buffer = '';
            clearLoading(loadingBubble);
            var assistantBubble = loadingBubble;
            while (true) {
                var result = await reader.read();
                if (result.done) break;
                buffer += decoder.decode(result.value, { stream: true });
                var events = buffer.split('\n\n');
                buffer = events.pop();
                for (var ei = 0; ei < events.length; ei++) {
                    var dataLine = findDataLine(events[ei]);
                    if (!dataLine) continue;
                    var parsed;
                    try { parsed = JSON.parse(dataLine.slice(5).trim()); } catch (_e) { continue; }
                    if (parsed.type === 'meta') {
                        currentCitations = Array.isArray(parsed.citations) ? parsed.citations : [];
                        currentInteractionId = parsed.interaction_id || null;
                        if (parsed.context_window) contextWindow = parsed.context_window;
                        if (parsed.visitor_id) {
                            visitorId = parsed.visitor_id;
                            try { localStorage.setItem(STORAGE_KEY_VISITOR, visitorId); } catch (_e) {}
                        }
                        if (parsed.conversation_id) {
                            conversationId = parsed.conversation_id;
                            try { localStorage.setItem(STORAGE_KEY_CONVERSATION, conversationId); } catch (_e) {}
                        }
                    } else if (parsed.type === 'token') {
                        assistantText += parsed.content || '';
                        assistantBubble.innerHTML = renderMarkdown(assistantText);
                        autoScroll();
                    } else if (parsed.type === 'error') {
                        if (assistantText) {
                            assistantBubble.innerHTML = renderMarkdown(assistantText);
                            var errNote = document.createElement('p');
                            errNote.style.cssText = 'color:#c0392b;font-size:0.8rem;margin-top:6px;';
                            errNote.textContent = parsed.message || 'An error occurred.';
                            assistantBubble.appendChild(errNote);
                        } else {
                            assistantBubble.className = 'msg msg-error';
                            assistantBubble.textContent = parsed.message || 'An error occurred.';
                        }
                        state.isLoading = false; setControls(false); restoreSuggestions(); return;
                    } else if (parsed.type === 'done') {
                        receivedDone = true;
                        if (parsed.prompt_tokens && contextWindow) {
                            updateContextBar(parsed.prompt_tokens, contextWindow);
                        }
                        commitDone(question, assistantText, assistantBubble);
                    }
                }
            }
            // Flush decoder and process any remaining buffered event
            var remaining = decoder.decode();
            if (remaining) buffer += remaining;
            if (buffer.trim()) {
                var tailData = findDataLine(buffer);
                if (tailData) {
                    try {
                        var tp = JSON.parse(tailData.slice(5).trim());
                        if (tp.type === 'token') {
                            assistantText += tp.content || '';
                            assistantBubble.innerHTML = renderMarkdown(assistantText);
                        } else if (tp.type === 'done' && !receivedDone) {
                            receivedDone = true;
                            if (tp.prompt_tokens && contextWindow) {
                                updateContextBar(tp.prompt_tokens, contextWindow);
                            }
                            commitDone(question, assistantText, assistantBubble);
                        }
                    } catch (_e) { /* ignore unparseable tail */ }
                }
            }
            // EOF without done: show warning, do not commit to history
            if (!receivedDone) {
                if (assistantText) {
                    assistantBubble.innerHTML = renderMarkdown(assistantText)
                        + '<p style="color:#e67e22;font-size:0.8rem;margin-top:6px;">Response may be incomplete.</p>';
                } else {
                    assistantBubble.className = 'msg msg-error';
                    assistantBubble.textContent = 'The assistant is temporarily unavailable.';
                }
                state.isLoading = false; setControls(false); restoreSuggestions();
            }
        } catch (_err) {
            clearLoading(loadingBubble);
            loadingBubble.className = 'msg msg-error';
            loadingBubble.textContent = 'Unable to connect. Please check your connection.';
            state.isLoading = false; setControls(false); restoreSuggestions();
        }
    }

    form.addEventListener('submit', function(e) { e.preventDefault(); sendMessage(textarea.value); });
})();
