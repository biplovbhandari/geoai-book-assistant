/* global Vue */
import { isRecord } from '../shared/format.js';

import {
    apiGet,
    apiPost,
    isAuthError,
    describeError,
} from '../shared/api.js';

const TEST_ENDPOINT = '/eval/prompt-test';
const NOT_FOUND = 404;

// ---- Response checks and pure helpers ----

/** True for a list of objects, which is what GET /prompts returns. */
function isRecordArray(value) {
    return Array.isArray(value) && value.every(isRecord);
}

/** True for a prompt detail: an object with its id and text. */
function isPromptDetail(value) {
    return isRecord(value) && value.id !== undefined && typeof value.content === 'string';
}

/** True for a test reply: an answer string, with sources and chunks as lists. */
function isTestResult(value) {
    return isRecord(value)
        && typeof value.answer === 'string'
        && Array.isArray(value.sources)
        && Array.isArray(value.chunks);
}

/** URL that activates one prompt version. The id is encoded so an unexpected value cannot change the path. */
function activateUrl(id) {
    return `/prompts/${encodeURIComponent(id)}/activate`;
}

/**
 * Builds the Prompt Lab state and actions.
 *
 * `session` is the shell's plain `{ token, id }` object: the token authenticates each request, and
 * the id lets a response that arrives after a disconnect be dropped. `onAuthError(reason)` is called
 * when the API rejects the token.
 */
export function usePromptLab(session, onAuthError) {
    const { ref } = Vue;

    const activePrompt = ref(null);
    const versions = ref([]);
    const editorContent = ref('');
    const labelInput = ref('');
    const testQuestion = ref('');
    const testResult = ref(null);
    const testing = ref(false);
    const saving = ref(false);
    const error = ref('');

    /**
     * Runs one request and routes its outcome. Returns the data on success, or undefined when the
     * session ended meanwhile, the token was rejected (the user is signed out) or the request failed.
     * `notFoundMessage` replaces the generic failure text for an HTTP 404.
     */
    async function call(failure, perform, notFoundMessage = '') {
        const sessionId = session.id;
        try {
            const data = await perform();
            if (sessionId !== session.id) {
                return undefined;
            }
            error.value = '';
            return data;
        } catch (err) {
            if (sessionId !== session.id) {
                return undefined;
            }
            if (isAuthError(err)) {
                onAuthError(err.detail ? `Sign-in failed: ${err.detail}` : 'Sign-in failed.');
                return undefined;
            }
            console.error(`Dashboard: ${failure}`, err);
            if (notFoundMessage && err.status === NOT_FOUND) {
                error.value = notFoundMessage;
            } else {
                error.value = `${failure} (${describeError(err)}).`;
            }
            return undefined;
        }
    }

    /** Loads the active prompt into the editor when `syncEditor` is true, and stores it either way. */
    async function loadActive(syncEditor) {
        const data = await call(
            'Could not load the active prompt',
            () => apiGet('/prompts/active', session.token, isPromptDetail)
        );
        if (data === undefined) {
            return false;
        }
        activePrompt.value = data;
        if (syncEditor) {
            editorContent.value = data.content;
        }
        return true;
    }

    /** Loads the list of all prompt versions. */
    async function loadVersions() {
        const data = await call(
            'Could not load prompt versions',
            () => apiGet('/prompts', session.token, isRecordArray)
        );
        if (data === undefined) {
            return false;
        }
        versions.value = data;
        return true;
    }

    /** Loads the active prompt and the version list in parallel. The editor takes the active text. */
    async function fetchAll() {
        await Promise.all([loadActive(true), loadVersions()]);
    }

    /** Runs the edited prompt against the test question. Does not save anything. */
    async function test() {
        if (testing.value || !editorContent.value.trim() || !testQuestion.value.trim()) {
            return;
        }
        testing.value = true;
        try {
            const data = await call(
                'Test failed',
                () => apiPost(
                    TEST_ENDPOINT,
                    session.token,
                    { system_prompt: editorContent.value, question: testQuestion.value },
                    isTestResult
                ),
                'Test endpoint not available yet.'
            );
            if (data !== undefined) {
                testResult.value = data;
            }
        } finally {
            testing.value = false;
        }
    }

    /** Saves the editor text as a new version, then refreshes the list and clears the label. */
    async function save() {
        if (saving.value || !editorContent.value.trim()) {
            return;
        }
        saving.value = true;
        try {
            const label = labelInput.value.trim();
            const created = await call(
                'Could not save the prompt',
                () => apiPost(
                    '/prompts',
                    session.token,
                    { content: editorContent.value, label: label || null },
                    (value) => isRecord(value) && value.id !== undefined
                )
            );
            if (created === undefined) {
                return;
            }
            labelInput.value = '';
            await Promise.all([loadActive(false), loadVersions()]);
        } finally {
            saving.value = false;
        }
    }

    /** Makes one version the active prompt, then reloads the active prompt (and the editor) and the list. */
    async function activate(id) {
        const activated = await call(
            'Could not activate the prompt',
            () => apiPost(activateUrl(id), session.token, {}, isRecord)
        );
        if (activated === undefined) {
            return;
        }
        await Promise.all([loadActive(true), loadVersions()]);
    }

    /** Clears all state. Called by the shell on disconnect. */
    function reset() {
        activePrompt.value = null;
        versions.value = [];
        editorContent.value = '';
        labelInput.value = '';
        testQuestion.value = '';
        testResult.value = null;
        testing.value = false;
        saving.value = false;
        error.value = '';
    }

    return {
        activePrompt, versions, editorContent, labelInput, testQuestion, testResult,
        testing, saving, error,
        fetchAll, test, save, activate, reset,
    };
}
