# 010 Performance and Scalability

## Hot Paths
Primary runtime hot path in [server/ws_server.py](../server/ws_server.py):
- `ws_amd` receive loop
- `process_final_audio`
- `silero_has_speech`
- `transcribe_audio`
- `classify_transcript`

These dominate request latency and compute usage.

## Expensive Operations
- Whisper transcription is the most expensive CPU/GPU operation.
- Silero VAD inference and model state reset per request.
- Frequent `np.concatenate()` in websocket chunk accumulation can cause repeated buffer reallocation.
- Audio encode/upload work adds CPU and network overhead when audio saving is enabled.

## Blocking and Event Loop Considerations
- The server offloads heavy inference to `asyncio.to_thread()`, reducing direct event-loop blocking.
- Some module-level initialization and helper startup happen synchronously at import time.
- The AGI client uses blocking socket calls with short timeouts and polling.

## Memory Behavior
- `raw_audio` grows with incoming frames and is repeatedly reallocated via concatenation.
- Large keyword lists in server memory are loaded once and reused.
- Model artifacts are loaded at startup and retained in memory.

## Concurrency Model
- FastAPI async websocket handlers
- Shared global model instances and threshold state
- Elasticsearch logger single-consumer queue worker
- Audio uploader daemon worker with retries

## Scalability Assumptions
- A single process can handle only bounded concurrent heavy transcriptions depending on GPU/CPU capacity.
- Horizontal scale is expected externally through multiple worker processes or instances.
- Each worker/process loads its own model copies.

## Caching Strategy
- No explicit request-level cache is implemented.
- Implicit reuse occurs through loaded models and in-memory keyword data.

## Bottlenecks and Risks (Verified)
1. Global runtime config mutation can cause cross-call interference under concurrency.
2. Repeated `np.concatenate()` may increase memory churn for long or large streams.
3. Elasticsearch queue growth can hide overload under sustained throughput.
4. Audio uploader queue backpressure can defer or drop persistence work when the target is slow.
5. Startup cold-load cost for models can affect first-request latency.

## Practical Optimization Opportunities
- Replace incremental concatenation with chunk-list buffering and a single final concat.
- Isolate mutable runtime parameters per connection.
- Add bounded frame-size validation to reduce accidental or malicious memory pressure.
- Tune worker counts and queue sizes using measured production load.
- Prewarm the model process before serving live traffic.
