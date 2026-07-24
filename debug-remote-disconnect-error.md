# Debug Session: remote-disconnect-error

- Status: [OPEN]
- Symptom: `full_test.py` repeatedly raises `('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))`.
- Expected: Model requests return an HTTP response.

## Hypotheses

1. The request URL, route, or protocol does not match the upstream API.
2. A local/system proxy or VPN is terminating the connection.
3. Request headers, authentication, or body format cause the gateway to close the connection.
4. Request rate, timeout, or connection reuse is rejected by the upstream gateway.
5. The model endpoint is unavailable, or TLS/network transport fails before a response is sent.

## Evidence

- Pre-fix logs show both non-stream and stream requests failed before any HTTP status with `ConnectionError(ProtocolError(...RemoteDisconnected...))`.
- No `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, or `NO_PROXY` environment variable was active.
- TCP port 443 was reachable.
- Python `requests` with its default User-Agent was disconnected, while browser/curl User-Agent values received HTTP responses.
- Post-fix, the original test no longer raises `RemoteDisconnected`; both calls receive HTTP 404.
- The HTTP 404 body is `model_not_found`: model `GPT-5.5` is not supported by any configured account in the current group.
- The duplicate error text is expected from `sample_metrics`: each sample sends one non-stream request and one stream request.

## Conclusion

- Confirmed hypothesis 3: the gateway silently drops the default Python Requests client signature.
- Rejected hypothesis 2: no environment proxy was active.
- Rejected hypothesis 4 for the reproduction: one short sample failed identically without load.
- Rejected hypothesis 5 at transport level: the endpoint returned structured HTTP responses after changing User-Agent.
- A separate configuration issue remains: `MODEL=GPT-5.5` is unavailable for the API key's current group.

## Changes

- Added a non-sensitive `User-Agent: OpenAI-Compatible-Test/1.0` request header.
- Added temporary network instrumentation for pre-fix/post-fix evidence. Instrumentation remains until user confirmation.
