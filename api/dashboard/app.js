/* global Vue */
import {
    formatLatency,
    formatDuration,
    formatDate,
    formatScore,
    formatSatisfaction,
    satClass,
    scoreClass,
    evalScoreClass,
    shortId,
    conversationTitle,
} from './shared/format.js';

import {
    readStoredToken,
    storeToken,
    clearStoredToken,
} from './shared/api.js';

import { useAnalytics } from './views/analytics.js';
import { useEval } from './views/eval.js';
import { usePromptLab } from './views/prompt-lab.js';
import { useRetrieval } from './views/retrieval.js';

if (typeof Vue === 'undefined') {
    // #app stays hidden (v-cloak) until Vue mounts, so a failed CDN load would otherwise leave a blank page.
    const bootError = document.getElementById('boot-error');
    bootError.textContent = 'Vue could not be loaded from the CDN. Check the network connection and reload.';
    bootError.hidden = false;
    // A module cannot `return` at the top level, so throwing is what stops the rest of it from running.
    throw new Error('Vue CDN unavailable');
}

const { createApp, ref, reactive, onMounted, onUnmounted } = Vue;

const REFRESH_MS = 60000;

createApp({
    setup() {
        const authenticated = ref(false);
        const tokenInput = ref('');
        const authError = ref('');
        // Plain variables on purpose: the template never reads them. The view composables share
        // `session` by reference to read the token and to tell whether a response is stale.
        const session = { token: '', id: 0 };
        let refreshTimer = null;

        const tab = ref('analytics');
        const analytics = useAnalytics(session, disconnect);
        const evaluation = useEval(session, disconnect);
        const promptLab = usePromptLab(session, disconnect);
        const retrieval = useRetrieval(session, disconnect);

        /** Stops the 60-second auto-refresh. */
        function stopRefresh() {
            if (refreshTimer !== null) {
                clearInterval(refreshTimer);
                refreshTimer = null;
            }
        }

        /**
         * Shows the dashboard, loads everything, and keeps the analytics and the evaluation run list
         * fresh until the user disconnects. The run list reloads whichever tab is open, so a run started
         * elsewhere (the CLI, or another browser) appears within a minute and, if it is still going, is
         * followed. The eval sets are not reloaded on the timer: they change only when a run starts, and
         * the refresh button on the evaluation tab reloads them.
         */
        function startSession() {
            session.id += 1;
            authenticated.value = true;
            analytics.fetchAll();
            evaluation.fetchAll();
            promptLab.fetchAll();
            stopRefresh();
            refreshTimer = setInterval(() => {
                analytics.fetchAll();
                evaluation.fetchRuns();
            }, REFRESH_MS);
        }

        /** Validates the typed token, saves it for this tab, and starts the session. */
        function connect() {
            const value = tokenInput.value.trim();
            if (!value) {
                authError.value = 'Enter the admin token.';
                return;
            }
            // fetch() throws an unhelpful TypeError for header values outside printable ASCII, and a
            // stray smart quote or zero-width space is easy to pick up when copying a token.
            if (!/^[\x20-\x7e]+$/.test(value)) {
                authError.value = 'The token contains characters that cannot be sent in an HTTP header.';
                return;
            }
            session.token = value;
            storeToken(value);
            tokenInput.value = '';
            authError.value = '';
            startSession();
        }

        /**
         * Ends the session: stops refreshing and following any running evaluation, forgets the token,
         * clears every section and the chart, and returns to the first tab.
         * `reason` is shown on the sign-in form; it is only a string when the server rejected the token.
         */
        function disconnect(reason) {
            session.id += 1;
            stopRefresh();
            analytics.cleanup();
            evaluation.cleanup();
            clearStoredToken();
            session.token = '';
            analytics.reset();
            evaluation.reset();
            promptLab.reset();
            retrieval.reset();
            tab.value = 'analytics';
            authError.value = typeof reason === 'string' ? reason : '';
            authenticated.value = false;
        }

        onMounted(() => {
            const stored = readStoredToken();
            if (stored) {
                session.token = stored;
                startSession();
            }
        });

        onUnmounted(() => {
            stopRefresh();
            analytics.cleanup();
            evaluation.cleanup();
        });

        return {
            tab,
            // reactive() unwraps each composable's refs, so the template reads `analytics.days` and
            // `eval.runs` rather than `.value` on every binding.
            analytics: reactive(analytics),
            // The local is `evaluation` because a strict-mode module cannot bind the name `eval`. As a
            // property name it is legal, and the template reads it as `eval`.
            eval: reactive(evaluation),
            promptLab: reactive(promptLab),
            retrieval: reactive(retrieval),
            // A string `ref` in the template is matched by name against the top level of this object,
            // so the chart canvas cannot live inside the analytics namespace.
            dailyCanvas: analytics.dailyCanvas,
            authenticated,
            tokenInput,
            authError,
            connect,
            disconnect,
            satClass,
            scoreClass,
            evalScoreClass,
            formatLatency,
            formatDuration,
            formatDate,
            formatScore,
            formatSatisfaction,
            shortId,
            conversationTitle,
        };
    },
}).mount('#app');
