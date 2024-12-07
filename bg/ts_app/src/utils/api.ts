class HTTPError extends Error {}
const apiHost = 'http://localhost:8080/api/v1';

function objToQueryString(obj) {
  const keyValuePairs = [];
  for (const key in obj) {
    keyValuePairs.push(
      encodeURIComponent(key) + '=' + encodeURIComponent(obj[key]),
    );
  }
  return keyValuePairs.join('&');
}
export const getURL = (endpoint: string | RequestInfo | URL) => apiHost + endpoint;

const makeQueryString = (url, obj) =>
  url + (obj == null ? '' : objToQueryString(obj));

const query = <T = unknown>(endpoint: RequestInfo | URL, init?: RequestInit) =>
  fetch(getURL(endpoint), {
   ...init,
  credentials: "include",
  }
       ).then(res => {
    if (!res.ok) throw new HTTPError(res.statusText, {cause: res});

    return res.json() as Promise<T>; // <--- Applying the generic type above
  });

const makeRequest =
  // -----------\/ RequestInit['method'] is a union of all the possible HTTP methods

    (method: RequestInit['method']) =>
    <TResponse = unknown, TBody = Record<string, unknown>>(
      url: RequestInfo | URL,
      body: TBody | null = null,
    ) => {
      if (method == 'GET') {
        return query<TResponse>(makeQueryString(url, body), {
          method,
        });
      }
      return query<TResponse>(url, {
        method,
        body: body == null ? null : JSON.stringify(body), // <-- JSON Stringify any given object
      });
    };

export const getHTML = (endpoint: RequestInfo | URL, init?: RequestInit) =>
  fetch(apiHost + endpoint, init).then(res => {
    if (!res.ok) throw new HTTPError(res.statusText, {cause: res});
    return res.text();
  });

export const get = makeRequest('GET');
export const post = makeRequest('POST');
export const push = makeRequest('PUSH');
