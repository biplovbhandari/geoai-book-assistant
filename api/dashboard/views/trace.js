/* global Vue */
import { isRecord, isNumber, formatCount } from '../shared/format.js';
import { apiGet, apiPost, isAuthError, describeError } from '../shared/api.js';

/** How many interactions to load per page of the history list. */
const PAGE_SIZE = 20;

/** Timeout for the LLM-backed trace and groundedness calls, which can run for a minute or more. */
const LLM_TIMEOUT_MS = 120000;

/** Stage name to scoring method label, shown in the chunk journey column headers. */
const SCORE_TYPE_LABELS = {
    dense: 'cosine',
    sparse: 'splade',
    fusion: 'RRF',
    reranking: 'logit',
};

/** Stage name to CSS class used for waterfall bar segment coloring. */
const STAGE_CSS_CLASSES = {
    dense: 'dense',
    sparse: 'sparse',
    fusion: 'fusion',
    reranking: 'rerank',
};

/**
 * Returns the scoring method label for a retrieval stage.
 * @param {string} stage - Stage name like 'dense', 'sparse', 'fusion', 'reranking'.
 * @returns {string} Human-readable score type like 'cosine', 'splade', 'RRF', 'logit'.
 */
function stageScoreType(stage) {
    return SCORE_TYPE_LABELS[stage] || '';
}

/**
 * Formats a chunk score for the journey table display. Uses 2 decimal places
 * for values with absolute value >= 1, and 3 decimal places for smaller values.
 * @param {number} score - The numeric score to format.
 * @returns {string} Formatted score string, or '' for non-numeric input.
 */
function formatChunkScore(score) {
    if (typeof score !== 'number') return '';
    return Math.abs(score) >= 1 ? score.toFixed(2) : score.toFixed(3);
}

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
     * Groups the loaded interactions by conversation_id for the history list.
     * Turns within a group run oldest first; groups run newest first by their latest turn.
     */
    /** Maximum gap (ms) between consecutive interactions to stay in one group. */
    const CONVERSATION_GAP_MS = 30 * 60 * 1000;

    /**
     * Interactions grouped by conversation_id, then split when consecutive turns are
     * more than 30 minutes apart. This prevents all-time grouping when the widget
     * persists one conversation_id across browser sessions.
     */
    const groupedInteractions = computed(() => {
        const byConversation = new Map();
        for (const item of interactions.value) {
            if (!byConversation.has(item.conversation_id)) byConversation.set(item.conversation_id, []);
            byConversation.get(item.conversation_id).push(item);
        }
        const groups = [];
        for (const [conversationId, items] of byConversation) {
            const sorted = [...items].sort((a, b) => Date.parse(a.created_at) - Date.parse(b.created_at));
            let current = [sorted[0]];
            for (let i = 1; i < sorted.length; i++) {
                const gap = Date.parse(sorted[i].created_at) - Date.parse(sorted[i - 1].created_at);
                if (gap > CONVERSATION_GAP_MS) {
                    groups.push({ conversationId: conversationId + '-' + groups.length, turns: current, turnCount: current.length });
                    current = [];
                }
                current.push(sorted[i]);
            }
            groups.push({ conversationId: conversationId + '-' + groups.length, turns: current, turnCount: current.length });
        }
        const newest = (group) => Date.parse(group.turns[group.turns.length - 1].created_at);
        return groups.sort((a, b) => newest(b) - newest(a));
    });

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

    // ---- Stat tiles and summary metrics for the redesigned trace viewer ----

    /**
     * Key metrics displayed as stat tiles at the top of the trace viewer.
     * Returns an array of { label, value, good? } objects with pre-formatted values.
     */
    const statTiles = computed(() => {
        const trace = activeTrace.value;
        if (!trace) return [];

        // Total latency: prefer the interaction record in history mode, else sum retrieval + generation.
        let totalMs = null;
        if (mode.value === 'history' && traceDetail.value && traceDetail.value.interaction) {
            totalMs = traceDetail.value.interaction.latency_ms;
        }
        if (totalMs == null) {
            const retMs = (trace.retrieval && trace.retrieval.total_latency_ms) || 0;
            const genMs = (trace.generation && trace.generation.generation_ms) || 0;
            if (retMs || genMs) totalMs = retMs + genMs;
        }

        // Chunks retrieved: last retrieval stage's output count, or fallback to chunk_count.
        const stages = trace.retrieval && trace.retrieval.stages;
        let chunks = null;
        if (Array.isArray(stages) && stages.length > 0) {
            chunks = stages[stages.length - 1].output_count;
        }
        if (chunks == null && trace.retrieval) {
            chunks = trace.retrieval.chunk_count;
        }

        // Total tokens from generation trace.
        const gen = trace.generation || {};
        const prompt = isNumber(gen.prompt_tokens) ? gen.prompt_tokens : 0;
        const completion = isNumber(gen.completion_tokens) ? gen.completion_tokens : 0;
        const totalTokens = (prompt || completion) ? prompt + completion : null;

        // Generation time.
        const genMs = gen.generation_ms;

        // Groundedness score (available only after the check has been run).
        const gResult = groundednessResult.value;
        const gScore = gResult && gResult.summary && isNumber(gResult.summary.score)
            ? gResult.summary.score : null;

        /** Formats milliseconds as "Xs" or "Xms" depending on magnitude. */
        function fmtMs(ms) {
            if (!isNumber(ms)) return '-';
            return ms >= 1000 ? (ms / 1000).toFixed(1) + 's' : ms + 'ms';
        }

        return [
            { label: 'Total Latency', value: fmtMs(totalMs) },
            { label: 'Chunks Retrieved', value: isNumber(chunks) ? String(chunks) : '-' },
            { label: 'Tokens Used', value: isNumber(totalTokens) ? formatCount(totalTokens) : '-' },
            { label: 'Generation', value: fmtMs(genMs) },
            { label: 'Groundedness', value: isNumber(gScore) ? gScore.toFixed(1) : '-', good: isNumber(gScore) && gScore >= 0.8 },
        ];
    });

    /**
     * Waterfall bar segments for retrieval stages. Each entry has the stage name,
     * raw latency in ms, percentage width of the bar, and a CSS class for coloring.
     */
    const waterfallSegments = computed(() => {
        const trace = activeTrace.value;
        if (!trace || !trace.retrieval || !Array.isArray(trace.retrieval.stages)) return [];
        const total = trace.retrieval.total_latency_ms || 1;
        return trace.retrieval.stages.map((s) => ({
            stage: s.stage,
            latency: s.latency_ms || 0,
            pct: Math.max(((s.latency_ms || 0) / total) * 100, 0),
            cssClass: STAGE_CSS_CLASSES[s.stage] || s.stage,
        }));
    });

    /**
     * Builds a unified chunk journey across all retrieval stages.
     *
     * Returns an object with:
     * - stageNames: ordered list of stage name strings
     * - rows: sorted array of { chunkId, displayName, scores, survived, lastScore }
     *   with survived chunks first (by descending last-stage score), then eliminated
     * - stageRanges: { [stage]: { min, max } } for computing score intensity classes
     */
    const chunkJourney = computed(() => {
        const trace = activeTrace.value;
        const empty = { stageNames: [], rows: [], stageRanges: {} };
        if (!trace || !trace.retrieval || !Array.isArray(trace.retrieval.stages)) return empty;

        const stages = trace.retrieval.stages;
        if (!stages.length) return empty;
        const stageNames = stages.map((s) => s.stage);

        // Map chunk_id to source_path from citations for display names.
        const nameMap = {};
        const citations = activeCitations.value || [];
        for (const cit of citations) {
            if (cit.chunk_id && cit.source_path) {
                nameMap[cit.chunk_id] = cit.source_path;
            }
        }

        // Collect chunk scores across all stages.
        const chunkMap = new Map();
        for (const stage of stages) {
            if (!Array.isArray(stage.chunk_scores)) continue;
            for (const cs of stage.chunk_scores) {
                if (!cs || cs.chunk_id == null) continue;
                const id = String(cs.chunk_id);
                if (!chunkMap.has(id)) chunkMap.set(id, {});
                if (typeof cs.score === 'number') {
                    chunkMap.get(id)[stage.stage] = cs.score;
                }
            }
        }

        // Chunks present in the last stage's output are considered "survived".
        const lastStage = stages[stages.length - 1];
        const survivedIds = new Set();
        if (Array.isArray(lastStage.chunk_scores)) {
            for (const cs of lastStage.chunk_scores) {
                if (cs && cs.chunk_id != null) survivedIds.add(String(cs.chunk_id));
            }
        }

        // Build rows with display name, survival status, and per-stage scores.
        const rows = [];
        for (const [chunkId, scores] of chunkMap) {
            const survived = survivedIds.has(chunkId);
            const lastScore = typeof scores[lastStage.stage] === 'number' ? scores[lastStage.stage] : null;
            const displayName = nameMap[chunkId] || chunkId;
            rows.push({ chunkId, displayName, scores, survived, lastScore });
        }

        // Survived chunks first by descending last-stage score, then eliminated.
        rows.sort((a, b) => {
            if (a.survived !== b.survived) return a.survived ? -1 : 1;
            const aScore = typeof a.lastScore === 'number' ? a.lastScore : -Infinity;
            const bScore = typeof b.lastScore === 'number' ? b.lastScore : -Infinity;
            return bScore - aScore;
        });

        // Per-stage min/max for intensity class calculation.
        const stageRanges = {};
        for (const stage of stages) {
            if (!Array.isArray(stage.chunk_scores)) continue;
            const nums = stage.chunk_scores
                .map((cs) => cs && cs.score)
                .filter((v) => typeof v === 'number');
            if (nums.length) {
                stageRanges[stage.stage] = { min: Math.min(...nums), max: Math.max(...nums) };
            }
        }

        return { stageNames, rows, stageRanges };
    });

    /**
     * Returns a CSS intensity class ('i0' through 'i3') for a chunk score at a
     * given stage. The class maps to background color intensity in the journey table.
     * Quartile boundaries are computed from each stage's score range.
     * @param {number} score - The chunk's score at this stage.
     * @param {string} stageName - The retrieval stage name.
     * @returns {string} One of 'i0', 'i1', 'i2', 'i3'.
     */
    function scoreIntensity(score, stageName) {
        if (typeof score !== 'number') return 'i0';
        const ranges = chunkJourney.value.stageRanges;
        const range = ranges[stageName];
        if (!range || range.max === range.min) return 'i1';
        const normalized = (score - range.min) / (range.max - range.min);
        if (normalized >= 0.75) return 'i3';
        if (normalized >= 0.5) return 'i2';
        if (normalized >= 0.25) return 'i1';
        return 'i0';
    }

    // ---- Section summary metrics for collapsible headers ----

    /**
     * Summary metric for the Retrieval section header.
     * Shows total latency, candidate counts from source stages, and final selected count.
     */
    const retrievalMetric = computed(() => {
        const ret = activeTrace.value && activeTrace.value.retrieval;
        if (!ret) return '';
        const parts = [];
        if (isNumber(ret.total_latency_ms)) parts.push(ret.total_latency_ms + 'ms');
        const stages = ret.stages;
        if (Array.isArray(stages) && stages.length > 0) {
            const sources = stages.filter((s) => s.stage !== 'fusion' && s.stage !== 'reranking');
            const counts = sources.map((s) => s.output_count).filter((v) => isNumber(v));
            const lastOut = stages[stages.length - 1].output_count;
            if (counts.length && isNumber(lastOut)) {
                parts.push(counts.join('/') + ' candidates, ' + lastOut + ' selected');
            }
        }
        return parts.join(' - ');
    });

    /** Summary metric for the Context Assembly section header. */
    const contextMetric = computed(() => {
        const trace = activeTrace.value;
        if (!trace) return '';
        const ctx = trace.context || {};
        const gen = trace.generation || {};
        const parts = [];
        if (isNumber(ctx.chunk_count)) parts.push(ctx.chunk_count + ' chunks');
        if (isNumber(gen.prompt_tokens)) parts.push('~' + formatCount(gen.prompt_tokens) + ' tokens');
        return parts.join(', ');
    });

    /** Summary metric for the Generation section header. */
    const generationMetric = computed(() => {
        const gen = activeTrace.value && activeTrace.value.generation;
        if (!gen) return '';
        const parts = [];
        if (gen.model) parts.push(gen.model);
        if (isNumber(gen.generation_ms)) parts.push(formatCount(gen.generation_ms) + 'ms');
        return parts.join(' - ');
    });

    /** Summary metric for the Response section header. */
    const responseMetric = computed(() => {
        const gen = activeTrace.value && activeTrace.value.generation;
        if (!gen || !isNumber(gen.completion_tokens)) return '';
        return formatCount(gen.completion_tokens) + ' tokens';
    });

    /** Summary metric for the Citations section header. */
    const citationsMetric = computed(() => {
        const citations = activeCitations.value;
        if (!citations || !citations.length) return '';
        return citations.length + ' source' + (citations.length !== 1 ? 's' : '');
    });

    /** Summary metric for the Groundedness section header. */
    const groundednessMetric = computed(() => {
        const result = groundednessResult.value;
        if (!result || !result.summary) return '';
        const s = result.summary;
        const parts = [];
        if (isNumber(s.supported) && isNumber(s.total_claims)) {
            parts.push(s.supported + '/' + s.total_claims + ' supported');
        }
        if (isNumber(s.score)) {
            parts.push('(' + s.score.toFixed(1) + ')');
        }
        return parts.join(' ');
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
                LLM_TIMEOUT_MS,
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
                LLM_TIMEOUT_MS,
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

    /** Formats a retrieval stage's config dict as "key=value" pairs, or '-' when it is empty. */
    function formatStageConfig(config) {
        if (!isRecord(config)) return '-';
        const pairs = Object.entries(config).map(([key, value]) => `${key}=${value}`);
        return pairs.length ? pairs.join(', ') : '-';
    }

    /** Formats the min-max of a stage's chunk scores, or '-' when the stage has none. */
    function formatScoreRange(chunkScores) {
        if (!Array.isArray(chunkScores)) return '-';
        const scores = chunkScores.map((c) => c && c.score).filter((v) => typeof v === 'number');
        if (!scores.length) return '-';
        return `${Math.min(...scores).toFixed(3)} - ${Math.max(...scores).toFixed(3)}`;
    }

    /** Formats a raw history message count into a human-readable label. */
    function formatHistory(msgCount) {
        if (msgCount == null) return '-';
        if (msgCount === 0) return 'None';
        const exchanges = Math.floor(msgCount / 2);
        if (exchanges <= 0) return msgCount + (msgCount === 1 ? ' message' : ' messages');
        return exchanges + (exchanges === 1 ? ' prior exchange' : ' prior exchanges');
    }

    return {
        formatStageConfig, formatScoreRange, formatChunkScore,
        stageScoreType, scoreIntensity, formatHistory,
        mode, interactions, interactionsTotal, interactionsPage,
        selectedId, traceDetail, liveQuestion, liveResult,
        loading, error, groundednessResult, groundednessLoading,
        messagesExpanded,
        activeTrace, activeCitations, activeResponse, activeQuestion,
        hasTrace, canCheckGroundedness, groupedInteractions,
        statTiles, waterfallSegments, chunkJourney,
        retrievalMetric, contextMetric, generationMetric,
        responseMetric, citationsMetric, groundednessMetric,
        fetchInteractions, fetchTrace, runLiveTrace, checkGroundedness,
        isStageExpanded, toggleStage,
        reset,
    };
}
