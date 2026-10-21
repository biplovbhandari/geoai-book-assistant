/* global Vue */
import {
    isNumber,
    isRecord,
    formatCount,
    formatDate,
    formatScore,
} from '../shared/format.js';

import {
    apiGet,
    apiPost,
    isAuthError,
    describeError,
} from '../shared/api.js';

// A run takes minutes (each question needs a generation and two judge calls), so a check every
// 10 seconds shows progress without adding load to a server that is busy with the run itself.
const POLL_MS = 10000;

// A run in either of these statuses is still going. The runner creates runs as 'running'; the database
// also allows 'pending', so it counts the same way.
const ACTIVE_STATUSES = ['pending', 'running'];
const STATUS_FAILED = 'failed';

// The symbol that opens each run's label in the picker. An <option> cannot hold a coloured dot, so the
// state is shown by the character itself: an hourglass (U+23F3) while the run is going, a check mark
// (U+2714) when it finished and a cross (U+2718) when it failed.
const RUN_MARKS = { active: '⏳', completed: '✔', failed: '✘' };

// How many characters of a question the question picker shows. The results table has the whole text.

// The four scores the runner records for every question. `label` heads the table column, `avgLabel`
// names the summary card, and `hint` is the column header's tooltip.
const SCORE_COLUMNS = [
    {
        key: 'retrieval_precision',
        label: 'Precision',
        avgLabel: 'Avg precision',
        hint: 'Share of the retrieved chunks that come from an expected source',
    },
    {
        key: 'retrieval_recall',
        label: 'Recall',
        avgLabel: 'Avg recall',
        hint: 'Share of the expected sources found among the retrieved chunks',
    },
    {
        key: 'answer_similarity',
        label: 'Similarity',
        avgLabel: 'Avg similarity',
        hint: 'LLM judge rating (0-1) of how well the answer matches the expected answer',
    },
    {
        key: 'faithfulness',
        label: 'Faithfulness',
        avgLabel: 'Avg faithfulness',
        hint: 'LLM judge rating (0-1) of how well the retrieved chunks support the answer',
    },
];

// ---- Response checks and pure helpers ----

/** True for an array whose entries are all objects, so a template can read their fields safely. */
function isRecordArray(value) {
    return Array.isArray(value) && value.every(isRecord);
}

/** True for a run detail the table can draw: `results` is a list of objects that each carry `scores`. */
function isRunDetail(value) {
    return isRecord(value)
        && Array.isArray(value.results)
        && value.results.every((result) => isRecord(result) && isRecord(result.scores));
}

/** True for the reply to starting a run. It has to name the run, or there is nothing to follow. */
function isRunStarted(value) {
    return isRecord(value) && value.eval_run_id !== null && value.eval_run_id !== undefined;
}

/**
 * True for the questions of one eval set: a list of objects that each have their text as a string and
 * a key that is a string or null. The key is null for a question saved before keys existed.
 */
function isQuestionList(value) {
    return isRecordArray(value) && value.every(
        (question) => typeof question.question === 'string'
            && (question.question_key === null || typeof question.question_key === 'string')
    );
}

/** Mean of the numeric entries, or null when there are none. A missing score is skipped, not counted as 0. */
function average(values) {
    const numbers = values.filter(isNumber);
    if (numbers.length === 0) {
        return null;
    }
    return numbers.reduce((sum, value) => sum + value, 0) / numbers.length;
}

/** True while a run has not finished, going by the status string the API reports. */
function isActive(status) {
    return ACTIVE_STATUSES.includes(status);
}

/** "1 question" or "15 questions". */
function countLabel(count, noun) {
    return `${formatCount(count)} ${noun}${count === 1 ? '' : 's'}`;
}

/** "5 of 15 questions" when planned is known, otherwise "5 questions". */
function progressLabel(scored, planned) {
    if (isNumber(planned) && planned > 0) {
        return `${formatCount(scored)} of ${formatCount(planned)} questions`;
    }
    return countLabel(scored, 'question');
}

/** Upper-cases the first letter, so the status "running" reads "Running". */
function capitalize(text) {
    return text.charAt(0).toUpperCase() + text.slice(1);
}

/** URL of one run's detail endpoint. The id is encoded so an unexpected value cannot change the path. */
function runUrl(id) {
    return `/eval/runs/${encodeURIComponent(id)}`;
}

/** URL of one eval set's question list. The id is encoded so an unexpected value cannot change the path. */
function setQuestionsUrl(setId) {
    return `/eval/sets/${encodeURIComponent(setId)}/questions`;
}

/** Reduces one API result to what its table row and detail panel show. */
function toRow(result, index) {
    const trace = isRecord(result.trace) ? result.trace : {};
    return {
        // The question's YAML id identifies a row across runs. A result whose question was deleted
        // from the set has none, so its position stands in.
        key: result.question_key ? String(result.question_key) : `row-${index}`,
        question: result.question || '',
        scores: result.scores,
        response: result.response || '',
        expectedAnswer: result.expected_answer || '',
        chunks: Array.isArray(trace.chunks) ? trace.chunks.filter(isRecord) : [],
    };
}

/**
 * Owns everything the evaluation view shows: the run and eval-set lists, the selected run's results,
 * and the requests that fill them, including starting a run and following it until it finishes.
 * `session` is the shell's plain `{ token, id }` object: the token authenticates each request, and
 * the id lets a response that arrives after a disconnect be dropped. `onAuthError(reason)` is called
 * when the API rejects the token, so the shell can sign the user out.
 * Must run inside a component's setup(), because its watcher is tied to that component.
 */
export function useEval(session, onAuthError) {
    // Read the global here instead of at module level, for the reason given in useAnalytics.
    const { ref, reactive, computed, watch } = Vue;

    const runs = ref(null);
    const sets = ref(null);
    const selectedSetName = ref('');
    // The selected eval set's questions that can be run on their own, or null until they have loaded.
    const questions = ref(null);
    // Keys of the questions to run, which are their ids in the eval set file. An empty list runs every
    // question.
    const selectedQuestionIds = ref([]);
    const selectedRunId = ref(null);
    const runDetail = ref(null);
    // Id of the run being followed: one started from this page, or one the API already reports as in
    // progress when the run list loads (started from the CLI, or before a reload). It drives the
    // progress polling and the busy button. A run cut short by a server crash or restart keeps the
    // status 'running' in the database, so it is followed, and keeps the button busy, until that row
    // is corrected.
    const runningId = ref(null);
    // True while the request that starts a run is in flight, so a double click cannot start two runs.
    const starting = ref(false);
    // One message per kind of request, '' when the last one worked. `run` covers starting a run and
    // checking on it.
    const errors = reactive({ runs: '', sets: '', questions: '', detail: '', run: '' });
    // Rows the user has opened, by question key. Keys rather than positions keep a row open while the
    // table refreshes during a run.
    const expandedKeys = reactive(new Set());

    // Plain variable on purpose: the template never reads it.
    let pollTimer = null;

    const busy = computed(() => starting.value || runningId.value !== null);

    // Database id of the selected eval set, which the questions endpoint needs. It is null until the
    // sets have loaded, and it changes only when another set is picked, not when the list is reloaded.
    const selectedSetId = computed(() => {
        const evalSet = (sets.value || []).find((candidate) => candidate.name === selectedSetName.value);
        return evalSet ? evalSet.id : null;
    });

    const rows = computed(() => (runDetail.value ? runDetail.value.results.map(toRow) : []));

    const metrics = computed(() => {
        const scored = rows.value;
        return [
            { label: 'Questions', value: formatCount(scored.length) },
            ...SCORE_COLUMNS.map((column) => ({
                label: column.avgLabel,
                value: formatScore(average(scored.map((row) => row.scores[column.key]))),
            })),
        ];
    });

    // One line about the selected run: its status, how far it got, and the pipeline settings it ran with.
    const statusLine = computed(() => {
        const detail = runDetail.value;
        if (!detail) {
            return '';
        }
        const config = isRecord(detail.config) ? detail.config : {};
        const scored = detail.results.length;
        const planned = detail.planned_count;
        const progress = isNumber(planned) && planned > 0
            ? `${formatCount(scored)} of ${formatCount(planned)} questions scored`
            : `${countLabel(scored, 'question')} scored`;
        return [
            capitalize(String(detail.status || 'unknown')),
            progress,
            config.retrieval_strategy,
            isNumber(config.retrieval_top_k) ? `k=${config.retrieval_top_k}` : '',
            config.reranker_provider,
            config.llm_model,
        ].filter(Boolean).join(' · ');
    });

    const statusFailed = computed(() => Boolean(runDetail.value) && runDetail.value.status === STATUS_FAILED);

    /** Text for a list with nothing to show: its load error, "Loading...", or an empty-state note. */
    function listMessage(key, data, emptyText) {
        if (errors[key]) {
            return errors[key];
        }
        if (data === null) {
            return 'Loading...';
        }
        return data.length === 0 ? emptyText : '';
    }

    /** Text for the results area when there is no table: an error, "Loading...", or why it is empty. */
    function detailMessage() {
        if (errors.detail) {
            return errors.detail;
        }
        if (selectedRunId.value === null) {
            return '';
        }
        if (runDetail.value === null) {
            return 'Loading...';
        }
        if (runDetail.value.results.length === 0) {
            return isActive(runDetail.value.status)
                ? 'Waiting for the first result...'
                : 'No results were recorded for this run.';
        }
        return '';
    }

    const messages = computed(() => ({
        runs: listMessage('runs', runs.value, 'No evaluation runs yet.'),
        // The set picker has no loading text: it appears once the sets arrive.
        sets: errors.sets || (sets.value !== null && sets.value.length === 0 ? 'No eval sets found.' : ''),
        detail: detailMessage(),
    }));

    /**
     * Text for one run in the picker: a symbol for its state, when it started, how it stands, and where
     * it was started from ("cli" or "api"). A run that is still going shows progress ("3 of 15 questions")
     * when planned_count is available. A failed run says how many questions it scored first. When
     * planned_count is null (old runs), only the scored count is shown.
     */
    function runLabel(run) {
        const when = formatDate(run.created_at);
        const progress = progressLabel(run.questions_evaluated, run.planned_count);
        const source = run.source ? ` (${run.source})` : '';
        if (isActive(run.status) || run.eval_run_id === runningId.value) {
            return `${RUN_MARKS.active} ${when} - ${progress} running${source}`;
        }
        if (run.status === STATUS_FAILED) {
            return `${RUN_MARKS.failed} ${when} - failed after ${progress}${source}`;
        }
        return `${RUN_MARKS.completed} ${when} - ${progress}${source}`;
    }

    /** Text for one entry of the eval-set picker. */
    function setLabel(evalSet) {
        return `${evalSet.name} (${countLabel(evalSet.question_count, 'question')})`;
    }


    /** True when the row for this question key is open. */
    function isExpanded(key) {
        return expandedKeys.has(key);
    }

    /** Opens or closes the detail panel under a question row. */
    function toggleExpand(key) {
        if (expandedKeys.has(key)) {
            expandedKeys.delete(key);
        } else {
            expandedKeys.add(key);
        }
    }

    // ---- Loading data ----

    const neverOutdated = () => false;

    /**
     * Runs one request and routes its outcome. Returns the parsed data on success. Returns undefined
     * when there is nothing to apply: the session ended or `isOutdated()` turned true while the request
     * was in flight, the API rejected the token (the user is then signed out), or the request failed.
     * A failure is also recorded in errors[key] as "<failure> (<reason>)."
     */
    async function call(key, failure, perform, isOutdated = neverOutdated) {
        const sessionId = session.id;
        try {
            const data = await perform();
            if (sessionId !== session.id || isOutdated()) {
                return undefined;
            }
            errors[key] = '';
            return data;
        } catch (err) {
            if (sessionId !== session.id || isOutdated()) {
                return undefined;
            }
            if (isAuthError(err)) {
                onAuthError(err.detail ? `Sign-in failed: ${err.detail}` : 'Sign-in failed.');
                return undefined;
            }
            console.error(`Dashboard: ${failure}`, err);
            errors[key] = `${failure} (${describeError(err)}).`;
            return undefined;
        }
    }

    /** Loads the eval sets and picks the first one when nothing valid is selected yet. */
    async function fetchSets() {
        const data = await call(
            'sets',
            'Could not load eval sets',
            () => apiGet('/eval/sets', session.token, isRecordArray)
        );
        if (data === undefined) {
            return;
        }
        sets.value = data;
        if (!data.some((evalSet) => evalSet.name === selectedSetName.value)) {
            selectedSetName.value = data.length > 0 ? data[0].name : '';
        }
    }

    /** Loads the questions of one eval set into `questions`, unless another set has been selected since. */
    async function loadQuestions(setId) {
        const data = await call(
            'questions',
            'Could not load questions',
            () => apiGet(setQuestionsUrl(setId), session.token, isQuestionList),
            () => selectedSetId.value !== setId
        );
        if (data !== undefined) {
            // A question saved before keys existed has no id to limit a run to, so it is not offered.
            questions.value = data.filter((question) => question.question_key);
        }
    }

    /**
     * Loads the run list. The first load also opens the latest run, which the API lists first. A run
     * the API reports as still going is followed, so its progress shows even when it was started elsewhere.
     */
    async function fetchRuns() {
        const data = await call(
            'runs',
            'Could not load runs',
            () => apiGet('/eval/runs', session.token, isRecordArray)
        );
        if (data === undefined) {
            return;
        }
        runs.value = data;
        // Follow a run that is already in progress (started from the CLI, or before this page loaded).
        // A run this page is already following is left alone.
        const active = data.find((run) => isActive(run.status));
        if (active && runningId.value === null) {
            runningId.value = active.eval_run_id;
        }
        // Later refreshes keep whatever run the user is looking at.
        if (selectedRunId.value === null && runs.value.length > 0) {
            selectRun(runs.value[0].eval_run_id);
        }
    }

    /** Loads one run's results into `runDetail`, unless another run has been selected since. */
    async function loadDetail(id) {
        const detail = await call(
            'detail',
            'Could not load the run',
            () => apiGet(runUrl(id), session.token, isRunDetail),
            () => selectedRunId.value !== id
        );
        if (detail !== undefined) {
            runDetail.value = detail;
        }
    }

    /** Selects a run; the watcher below loads its results. */
    function selectRun(id) {
        selectedRunId.value = id;
    }

    // The selection is the single trigger for loading a run, so the picker only has to set it and
    // nothing depends on the order in which a <select> fires v-model and change handlers. The previous
    // run's table is cleared first, so it is not mistaken for the new run's while the request is out.
    watch(selectedRunId, (id) => {
        runDetail.value = null;
        errors.detail = '';
        if (id !== null) {
            loadDetail(id);
        }
    });

    // The eval set selection is likewise the single trigger for loading its questions. The previous set's
    // questions and the ones picked from them are dropped first, so an id that belongs to another
    // set is never sent with this one.
    watch(selectedSetId, (id) => {
        questions.value = null;
        selectedQuestionIds.value = [];
        errors.questions = '';
        if (id !== null) {
            loadQuestions(id);
        }
    });

    /** Loads the sets and the runs in parallel; one failing request does not stop the other. */
    function fetchAll() {
        return Promise.allSettled([fetchSets(), fetchRuns()]);
    }

    // ---- Starting a run and following it ----

    /** Cancels the pending progress check, if any. */
    function stopPolling() {
        if (pollTimer !== null) {
            clearTimeout(pollTimer);
            pollTimer = null;
        }
    }

    /**
     * Checks a running evaluation once. While it is the selected run its results are refreshed, so the
     * table fills in as questions finish. A run that is no longer running stops the polling, and the run
     * list is reloaded so the picker shows its final question count.
     */
    async function pollRun(id) {
        const detail = await call(
            'run',
            'Could not check run progress',
            () => apiGet(runUrl(id), session.token, isRunDetail),
            () => runningId.value !== id
        );
        // A failed check is not the end of the run. It is reported and tried again at the next interval.
        if (detail === undefined) {
            return;
        }
        if (selectedRunId.value === id) {
            runDetail.value = detail;
        }
        if (!isActive(detail.status)) {
            runningId.value = null;
            await fetchRuns();
        }
    }

    /**
     * Waits one interval, checks the run, and queues the next check while it is still the run being
     * followed. Each check starts only after the previous one ended, so a slow server never gets
     * overlapping requests.
     */
    function schedulePoll(id) {
        pollTimer = setTimeout(async () => {
            pollTimer = null;
            await pollRun(id);
            if (runningId.value === id) {
                schedulePoll(id);
            }
        }, POLL_MS);
    }

    watch(runningId, (id) => {
        stopPolling();
        if (id !== null) {
            schedulePoll(id);
        }
    });

    /**
     * Starts an evaluation of the named set through POST /eval/runs and follows it until it finishes.
     * `tags` optionally narrows the run to questions with those tags, and `questionIds` to the questions
     * with those ids in the eval set file. An empty or missing list means no filter. Does nothing while
     * a run is already being started or followed. The server runs the questions in the background, so
     * the reply only says that the run began.
     */
    async function startRun(evalSetName, tags, questionIds) {
        if (busy.value || !evalSetName) {
            return;
        }
        starting.value = true;
        errors.run = '';
        try {
            // The API rejects unknown fields, so these names must match its request model exactly.
            const body = { eval_set_name: evalSetName };
            if (Array.isArray(tags) && tags.length > 0) {
                body.tags = tags;
            }
            // No selection means every question, so the field is left out rather than sent empty.
            if (Array.isArray(questionIds) && questionIds.length > 0) {
                body.question_ids = questionIds;
            }
            const started = await call(
                'run',
                'Could not start the run',
                () => apiPost('/eval/runs', session.token, body, isRunStarted)
            );
            if (started === undefined) {
                return;
            }
            if (isActive(started.status)) {
                runningId.value = started.eval_run_id;
            }
            // Select the new run first so the list reload that follows does not pick a run on its own.
            selectRun(started.eval_run_id);
            await fetchRuns();
        } finally {
            starting.value = false;
        }
    }

    // ---- CSV export ----

    /**
     * Downloads the selected run's results as a CSV file. The export endpoint returns the file directly,
     * so the response is fetched as a blob and offered as a download. Does nothing when no run is selected.
     */
    async function downloadCsv() {
        var id = selectedRunId.value;
        if (id === null) {
            return;
        }
        try {
            var response = await fetch('/eval/runs/' + encodeURIComponent(id) + '/export', {
                headers: { 'Authorization': 'Bearer ' + session.token },
            });
            if (!response.ok) {
                var text = await response.text().catch(function () { return ''; });
                errors.detail = 'Export failed (' + (text || response.status) + ').';
                return;
            }
            var blob = await response.blob();
            var url = URL.createObjectURL(blob);
            var a = document.createElement('a');
            a.href = url;
            a.download = 'eval_run_' + id + '.csv';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } catch (err) {
            console.error('Dashboard: CSV export failed', err);
            errors.detail = 'Export failed (' + describeError(err) + ').';
        }
    }

    // ---- Lifecycle ----

    /** Clears all data, errors and the run being followed. Called by the shell on disconnect. */
    function reset() {
        stopPolling();
        runs.value = null;
        sets.value = null;
        selectedSetName.value = '';
        questions.value = null;
        selectedQuestionIds.value = [];
        selectedRunId.value = null;
        runDetail.value = null;
        runningId.value = null;
        starting.value = false;
        expandedKeys.clear();
        Object.keys(errors).forEach((key) => {
            errors[key] = '';
        });
    }

    /** Stops following a running evaluation. Called by the shell on unmount and disconnect. */
    function cleanup() {
        stopPolling();
        // Without this, a check still in flight would queue another one when it finishes.
        runningId.value = null;
    }

    return {
        runs, sets, selectedSetName, questions, selectedQuestionIds, selectedRunId, runDetail, runningId,
        errors, busy, rows, metrics, statusLine, statusFailed, messages,
        scoreColumns: SCORE_COLUMNS,
        fetchAll, fetchRuns, fetchSets, selectRun, startRun, downloadCsv,
        runLabel, setLabel, isExpanded, toggleExpand,
        reset,
        cleanup,
    };
}
