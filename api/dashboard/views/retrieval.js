/* global Vue */
import { isRecord } from '../shared/format.js';
import { apiPost, isAuthError, describeError } from '../shared/api.js';

/** True for a /debug/query reply: an object holding the chunk list. */
function isInspectResult(value) {
    return isRecord(value) && Array.isArray(value.chunks);
}

/** Splits the comma-separated prefix text into trimmed, non-empty prefixes. */
function parseSources(text) {
    return text.split(',').map((part) => part.trim()).filter((part) => part !== '');
}

const PREVIEW_CHARS = 200;

/** Adds `textPreview` (the first PREVIEW_CHARS characters of `text`) to each chunk of a reply. */
function withPreviews(data) {
    return {
        ...data,
        chunks: data.chunks.map((chunk) => {
            const text = typeof chunk.text === 'string' ? chunk.text : '';
            return { ...chunk, text, textPreview: text.slice(0, PREVIEW_CHARS) };
        }),
    };
}

/**
 * Owns the Retrieval Inspector view: the question and expected-source inputs, and the retrieval result
 * for the last inspection. `session` is the shell's plain `{ token, id }` object, and
 * `onAuthError(reason)` signs the user out when the API rejects the token.
 * Must run inside a component's setup().
 */
export function useRetrieval(session, onAuthError) {
    const { ref, reactive } = Vue;

    const question = ref('');
    // Comma-separated source path prefixes; chunks starting with one are highlighted.
    const expectedSources = ref('');
    const results = ref(null);
    const loading = ref(false);
    const error = ref('');
    // Chunk ranks whose full text is shown. Cleared whenever new results arrive.
    const expandedRanks = reactive(new Set());

    /** True when the chunk at this rank shows its full text. */
    function isExpanded(rank) {
        return expandedRanks.has(rank);
    }

    /** Switches a chunk row between its preview and its full text. */
    function toggleExpand(rank) {
        if (expandedRanks.has(rank)) {
            expandedRanks.delete(rank);
        } else {
            expandedRanks.add(rank);
        }
    }

    /** Runs retrieval for the question and stores the response, or an error message. */
    async function inspect() {
        const text = question.value.trim();
        if (text === '' || loading.value) {
            return;
        }
        const sessionId = session.id;
        const body = { question: text };
        const sources = parseSources(expectedSources.value);
        if (sources.length > 0) {
            body.expected_sources = sources;
        }
        loading.value = true;
        error.value = '';
        try {
            const data = await apiPost('/debug/query', session.token, body, isInspectResult);
            if (sessionId !== session.id) {
                return;
            }
            expandedRanks.clear();
            results.value = withPreviews(data);
        } catch (err) {
            if (sessionId !== session.id) {
                return;
            }
            if (isAuthError(err)) {
                onAuthError(err.detail ? `Sign-in failed: ${err.detail}` : 'Sign-in failed.');
                return;
            }
            console.error('Dashboard: inspecting the query failed', err);
            error.value = `Could not inspect the query (${describeError(err)}).`;
        } finally {
            if (sessionId === session.id) {
                loading.value = false;
            }
        }
    }

    /** Clears the inputs, the result and any error. */
    function reset() {
        question.value = '';
        expectedSources.value = '';
        results.value = null;
        loading.value = false;
        error.value = '';
        expandedRanks.clear();
    }

    return {
        question, expectedSources, results, loading, error,
        inspect, reset, toggleExpand, isExpanded,
    };
}
