/* global Vue */
import { isRecord } from '../shared/format.js';
import { apiGet, apiPost, isAuthError, describeError } from '../shared/api.js';

/** How many interactions to load per page of the history list. */
const PAGE_SIZE = 20;

/** True for a GET /debug/interactions response: an object with an interactions array and total. */
function isInteractionList(value) {
    return isRecord(value) && Array.isArray(value.interactions);
}

/** True for a GET /debug/traces/{id} response: an object with interaction dict, traces dict, and citations. */
function isTraceDetail(value) {
    return isRecord(value) && isRecord(value.interaction) && isRecord(value.traces);
}

/** True for a POST /debug/trace response: an object with an answer string and a trace dict. */
function isLiveTrace(value) {
    return isRecord(value) && typeof value.answer === 'string' && isRecord(value.trace);
}

/** True for a POST /debug/groundedness response: an object with a claims array and a summary dict. */
function isGroundednessResult(value) {
    return isRecord(value) && Array.isArray(value.claims) && isRecord(value.summary);
}

/**
 * Owns the Pipeline Trace view: the History and Live modes, the shared trace viewer with
 * collapsible stages, and the groundedness check.
 *
 * `session` is the shell's plain `{ token, id }` object: the token authenticates each request,
 * and the id lets a response that arrives after a disconnect be dropped. `onAuthError(reason)`
 * is called when the API rejects the token, so the shell can sign the user out.
 *
 * Must run inside a component's setup().
 */
export function useTrace(session, onAuthError) {
    const { ref, reactive, computed } = Vue;

    const mode = ref('history');

    // History mode
    const interactions = ref([]);
    const interactionsTotal = ref(0);
    const interactionsPage = ref(0);
    const selectedId = ref(null);
    const traceDetail = ref(null);

    // Live mode
    const liveQuestion = ref('');
    const liveResult = ref(null);

    // Shared
    const loading = ref(false);
    const error = ref('');

    // Groundedness
    const groundednessResult = ref(null);
    const groundednessLoading = ref(false);

    // Collapsible sections in the trace viewer, tracked by section name.
    const expandedStages = reactive(new Set());

    // Whether context messages are shown inside the Context Assembly section.
    const messagesExpanded = ref(false);

    // ---- Unified view for the trace viewer ----

    /** The trace dict (retrieval, context, generation) from whichever mode is active, or null. */
    const activeTrace = computed(() => {
        if (mode.value === 'history' && traceDetail.value) return traceDetail.value.traces;
        if (mode.value === 'live' && liveResult.value) return liveResult.value.trace;
        return null;
    });

    /** The citations array from whichever mode is active. */
    const activeCitations = computed(() => {
        if (mode.value === 'history' && traceDetail.value) return traceDetail.value.citations;
        if (mode.value === 'live' && liveResult.value) return liveResult.value.citations;
        return [];
    });

    /** The response text from whichever mode is active. */
    const activeResponse = computed(() => {
        if (mode.value === 'history' && traceDetail.value) return traceDetail.value.interaction.response_text;
        if (mode.value === 'live' && liveResult.value) return liveResult.value.answer;
        return '';
    });

    /** The question text from whichever mode is active. */
    const activeQuestion = computed(() => {
        if (mode.value === 'history' && traceDetail.value) return traceDetail.value.interaction.question;
        if (mode.value === 'live') return liveQuestion.value;
        return '';
    });

    /** True when a trace is loaded and the viewer should be shown. */
    const hasTrace = computed(() => activeTrace.value !== null);

    /**
     * True when the active mode has enough data for a groundedness check: a response text
     * and at least one chunk with content.
     */
    const canCheckGroundedness = computed(() => {
        if (!hasTrace.value) return false;
        if (mode.value === 'live' && liveResult.value) {
            return Boolean(liveResult.value.answer)
                && Array.isArray(liveResult.value.sources)
                && liveResult.value.sources.length > 0;
        }
        if (mode.value === 'history' && traceDetail.value) {
            return Boolean(traceDetail.value.interaction.response_text)
                && Array.isArray(traceDetail.value.citations)
                && traceDetail.value.citations.length > 0;
        }
        return false;
    });

    // ---- Collapsible sections ----

    /** True when the trace section with this name is expanded. */
    function isStageExpanded(name) {
        return expandedStages.has(name);
    }

    /** Opens or closes a trace section by name. */
    function toggleStage(name) {
        if (expandedStages.has(name)) {
            expandedStages.delete(name);
        } else {
            expandedStages.add(name);
        }
    }

    // ---- API calls ----

    /**
     * Loads one page of past interactions into the history list.
     * @param {number} offset - Zero-based offset into the full list.
     */
    async function fetchInteractions(offset = 0) {
        const sessionId = session.id;
        loading.value = true;
        error.value = '';
        try {
            const data = await apiGet(
                `/debug/interactions?limit=${PAGE_SIZE}&offset=${offset}`,
                session.token,
                isInteractionList,
            );
            if (sessionId !== session.id) return;
            interactions.value = data.interactions;
            interactionsTotal.value = data.total;
            interactionsPage.value = offset;
        } catch (err) {
            if (sessionId !== session.id) return;
            if (isAuthError(err)) {
                onAuthError(err.detail ? `Sign-in failed: ${err.detail}` : 'Sign-in failed.');
                return;
            }
            console.error('Dashboard: loading interactions failed', err);
            error.value = `Could not load interactions (${describeError(err)}).`;
        } finally {
            if (sessionId === session.id) loading.value = false;
        }
    }

    /**
     * Loads the full trace detail for one past interaction.
     * Clears any previous groundedness result and collapses all sections.
     * @param {string} interactionId - UUID of the interaction to load.
     */
    async function fetchTrace(interactionId) {
        const sessionId = session.id;
        loading.value = true;
        error.value = '';
        groundednessResult.value = null;
        expandedStages.clear();
        messagesExpanded.value = false;
        try {
            const data = await apiGet(
                `/debug/traces/${encodeURIComponent(interactionId)}`,
                session.token,
                isTraceDetail,
            );
            if (sessionId !== session.id) return;
            selectedId.value = interactionId;
            traceDetail.value = data;
        } catch (err) {
            if (sessionId !== session.id) return;
            if (isAuthError(err)) {
                onAuthError(err.detail ? `Sign-in failed: ${err.detail}` : 'Sign-in failed.');
                return;
            }
            console.error('Dashboard: loading trace failed', err);
            error.value = `Could not load trace (${describeError(err)}).`;
        } finally {
            if (sessionId === session.id) loading.value = false;
        }
    }

    /**
     * Runs the full pipeline for the live question and stores the answer with its trace.
     * Does nothing when no question is entered or a request is already in flight.
     */
    async function runLiveTrace() {
        const text = liveQuestion.value.trim();
        if (!text || loading.value) return;
        const sessionId = session.id;
        loading.value = true;
        error.value = '';
        groundednessResult.value = null;
        expandedStages.clear();
        messagesExpanded.value = false;
        try {
            const data = await apiPost(
                '/debug/trace',
                session.token,
                { question: text },
                isLiveTrace,
            );
            if (sessionId !== session.id) return;
            liveResult.value = data;
        } catch (err) {
            if (sessionId !== session.id) return;
            if (isAuthError(err)) {
                onAuthError(err.detail ? `Sign-in failed: ${err.detail}` : 'Sign-in failed.');
                return;
            }
            console.error('Dashboard: live trace failed', err);
            error.value = `Live trace failed (${describeError(err)}).`;
        } finally {
            if (sessionId === session.id) loading.value = false;
        }
    }

    /**
     * Runs a groundedness check against the active trace's response and chunks.
     *
     * In History mode the chunks come from the stored citations, which include the full content
     * via the ChunkRecord join. In Live mode the chunks come from the pipeline's source output.
     *
     * Does nothing when the data needed for a check is not available.
     */
    async function checkGroundedness() {
        let responseText;
        let chunks;

        if (mode.value === 'live' && liveResult.value) {
            responseText = liveResult.value.answer;
            chunks = (liveResult.value.sources || []).map((s) => ({
                source_path: (s.chunk && s.chunk.metadata && s.chunk.metadata.source_path) || 'unknown',
                content: (s.chunk && s.chunk.content) || '',
            }));
        } else if (mode.value === 'history' && traceDetail.value) {
            responseText = traceDetail.value.interaction.response_text;
            chunks = (traceDetail.value.citations || []).map((c) => ({
                source_path: c.source_path || 'unknown',
                content: c.content || '',
            }));
        } else {
            return;
        }

        if (!responseText || chunks.length === 0) return;

        const sessionId = session.id;
        groundednessLoading.value = true;
        error.value = '';
        try {
            const data = await apiPost(
                '/debug/groundedness',
                session.token,
                { response_text: responseText, chunks },
                isGroundednessResult,
            );
            if (sessionId !== session.id) return;
            groundednessResult.value = data;
        } catch (err) {
            if (sessionId !== session.id) return;
            if (isAuthError(err)) {
                onAuthError(err.detail ? `Sign-in failed: ${err.detail}` : 'Sign-in failed.');
                return;
            }
            console.error('Dashboard: groundedness check failed', err);
            error.value = `Groundedness check failed (${describeError(err)}).`;
        } finally {
            if (sessionId === session.id) groundednessLoading.value = false;
        }
    }

    // ---- Lifecycle ----

    /** Clears all state. Called by the shell on disconnect. */
    function reset() {
        mode.value = 'history';
        interactions.value = [];
        interactionsTotal.value = 0;
        interactionsPage.value = 0;
        selectedId.value = null;
        traceDetail.value = null;
        liveQuestion.value = '';
        liveResult.value = null;
        loading.value = false;
        error.value = '';
        groundednessResult.value = null;
        groundednessLoading.value = false;
        expandedStages.clear();
        messagesExpanded.value = false;
    }

    return {
        mode, interactions, interactionsTotal, interactionsPage,
        selectedId, traceDetail, liveQuestion, liveResult,
        loading, error, groundednessResult, groundednessLoading,
        messagesExpanded,
        activeTrace, activeCitations, activeResponse, activeQuestion,
        hasTrace, canCheckGroundedness,
        fetchInteractions, fetchTrace, runLiveTrace, checkGroundedness,
        isStageExpanded, toggleStage,
        reset,
    };
}
