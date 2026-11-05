const TOKEN_KEY = 'dashboard_token';
const REQUEST_TIMEOUT_MS = 15000;
const AUTH_STATUSES = [401, 403];

// ---- Token storage ----
// sessionStorage can throw (blocked storage, some private modes). The dashboard still works for the
// current page load in that case; it only asks for the token again after a reload.

/** Returns the saved admin token, or '' when there is none or storage is unavailable. */
export function readStoredToken() {
    try {
        return sessionStorage.getItem(TOKEN_KEY) || '';
    } catch (_e) {
        return '';
    }
}

/** Saves the admin token for this tab; sessionStorage clears it when the tab closes. */
export function storeToken(value) {
    try {
        sessionStorage.setItem(TOKEN_KEY, value);
    } catch (_e) {
        // Storage unavailable: the token lives in memory for this page load only.
    }
}

/** Removes the saved admin token. */
export function clearStoredToken() {
    try {
        sessionStorage.removeItem(TOKEN_KEY);
    } catch (_e) {
        // Nothing was stored, so there is nothing to clear.
    }
}

// ---- API access ----

/** An HTTP failure from the analytics API, keeping the status code and FastAPI's `detail` text. */
export class ApiError extends Error {
    constructor(status, detail) {
        super(detail ? `HTTP ${status}: ${detail}` : `HTTP ${status}`);
        this.name = 'ApiError';
        this.status = status;
        this.detail = detail;
    }
}

/** Reads FastAPI's {"detail": "..."} error body, or '' when the body is absent or shaped differently. */
export async function readErrorDetail(response) {
    try {
        const body = await response.json();
        return typeof body.detail === 'string' ? body.detail : '';
    } catch (_e) {
        return '';
    }
}

/**
 * Sends one request with the Bearer token and returns the parsed JSON. apiGet and apiPost both go
 * through here, so the timeout, the error handling and the shape check live in one place.
 * `init` holds the fetch options that differ per method (method, body, extra headers).
 * `timeoutMs` overrides the default timeout for slow, LLM-backed endpoints.
 */
async function request(url, token, isValid, init, timeoutMs = REQUEST_TIMEOUT_MS) {
    const response = await fetch(url, {
        ...init,
        headers: { Accept: 'application/json', Authorization: `Bearer ${token}`, ...init.headers },
        cache: 'no-store',
        signal: AbortSignal.timeout(timeoutMs),
    });
    if (!response.ok) {
        throw new ApiError(response.status, await readErrorDetail(response));
    }
    const data = await response.json();
    if (!isValid(data)) {
        throw new ApiError(response.status, 'Unexpected response shape');
    }
    return data;
}

/**
 * GETs one analytics endpoint with the Bearer token and returns the parsed JSON.
 * Throws ApiError for a non-2xx status or a body that fails `isValid`. Validating the shape here
 * keeps a malformed response from reaching the templates, which assume arrays and objects.
 */
export function apiGet(url, token, isValid) {
    return request(url, token, isValid, {});
}

/**
 * POSTs `body` as JSON to one endpoint with the Bearer token and returns the parsed JSON reply.
 * Fails the same way as apiGet. The caller must not retry on its own: a POST is not safe to repeat.
 * `timeoutMs` is optional and defaults to the shared request timeout.
 */
export function apiPost(url, token, body, isValid, timeoutMs) {
    return request(url, token, isValid, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    }, timeoutMs);
}

/** Turns a failed request into short text for the section that could not load. */
export function describeError(err) {
    if (err instanceof ApiError) {
        return err.message;
    }
    if (err && err.name === 'TimeoutError') {
        return 'request timed out';
    }
    if (err instanceof SyntaxError) {
        return 'invalid response';
    }
    return 'network error';
}

/** True when the API rejected the admin token (HTTP 401 or 403), which is the cue to sign the user out. */
export function isAuthError(err) {
    return err instanceof ApiError && AUTH_STATUSES.includes(err.status);
}
