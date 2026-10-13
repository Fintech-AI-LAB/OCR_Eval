"""Replaceable LLM transport: request an API and return a normalized JSON result.

Adapters must return ``{"output": <parsed JSON object>}``. Optional metadata:
response_id, model, usage, request_sha256. Refused or incomplete responses must
raise an error instead of returning extraction decisions.
"""
import hashlib
import json
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, build_opener, HTTPRedirectHandler


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Do not forward bearer credentials to redirect destinations.


def validate_base_url(base_url):
    parsed = urlparse(base_url)
    if parsed.scheme != 'https' and not (parsed.scheme == 'http' and parsed.hostname in {'localhost', '127.0.0.1', '::1'}):
        raise ValueError('Use HTTPS, or HTTP on localhost.')
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('API base URL cannot contain credentials, query or fragment.')


def call_llm_api(*, model, base_url, api_key, system_prompt, payload,
                 timeout=90, retries=2, opener=None):
    """Call an OpenAI-compatible endpoint and normalize its completed response.

    Replace this function, or pass another callable as ChatClient(request_fn=...),
    to use a different provider protocol. The callback receives these same keyword
    arguments and owns authentication, provider-specific parsing and retries.
    No network request is made until this function is called.
    """
    validate_base_url(base_url)
    if not api_key or not model:
        raise ValueError('Set OPENAI_API_KEY and provide --model (or OPENAI_MODEL).')
    if type(retries) is not int or retries < 0:
        raise ValueError('retries must be a nonnegative integer.')
    body = {'model': model, 'response_format': {'type': 'json_object'},
            'messages': [{'role': 'system', 'content': system_prompt},
                         {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]}
    request = Request(base_url.rstrip('/') + '/chat/completions',
                      data=json.dumps(body).encode(),
                      headers={'Authorization': 'Bearer ' + api_key, 'Content-Type': 'application/json'})
    opener = opener if opener is not None else build_opener(NoRedirect())
    for attempt in range(retries + 1):
        try:
            with opener.open(request, timeout=timeout) as response:
                result = json.load(response)
            break
        except HTTPError as error:
            if error.code in {429, 500, 502, 503, 504} and attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise ValueError(f'LLM API HTTP {error.code}; check endpoint, model and credentials.') from None
        except (URLError, TimeoutError):
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise ValueError('LLM API connection failed or timed out.') from None
    if not isinstance(result, dict) or not isinstance(result.get('choices'), list) or not result['choices']:
        raise ValueError('LLM API returned no completion choices.')
    choice = result['choices'][0]
    if not isinstance(choice, dict) or not isinstance(choice.get('message'), dict):
        raise ValueError('LLM API returned an invalid completion message.')
    message = choice['message']
    if choice.get('finish_reason') != 'stop' or message.get('refusal'):
        raise ValueError('LLM refused or returned an incomplete response; no fields accepted.')
    try:
        content = json.loads(message.get('content') or '')
    except (TypeError, json.JSONDecodeError):
        raise ValueError('LLM response must be a JSON object.') from None
    if not isinstance(content, dict):
        raise ValueError('LLM response must be a JSON object.')
    request_hash = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False,
                                             allow_nan=False).encode()).hexdigest()
    return {'output': content, 'request_sha256': request_hash,
            'response_id': result.get('id'), 'model': result.get('model'),
            'usage': result.get('usage')}
