# Images Plan — `07-inference-systems`

| Asset | Placement | Teaching job |
|---|---|---|
| `kv-cache-mechanism.png` | Part 1 intro | Step-by-step: prompt tokens run once (prefill); each new token appends one K/V pair (decode); cached K/Vs shown as a growing teal list |
| `continuous-batching-vs-static.png` | Part 2 intro | Timeline: static batching idles while short requests wait for long ones; continuous batching slots in new requests as soon as a slot frees |
| `speculative-decoding-accept-reject.png` | Part 4 intro | Draft model proposes 5 tokens; verifier accepts 3, rejects 2; one verifier call replaces 3 sequential decode calls |

```text
[kv-cache-mechanism.png]
Flat vector sequence diagram, wide 16:9, dark graphite background. Left: "Prefill"
phase showing all prompt tokens processed in parallel (teal arrows to K/V cache).
Right: "Decode" phase showing one new token generated per step, appending one K/V
pair (amber arrow) to the growing teal cache list. The cache grows step by step.
Ivory labels. No logos, no photorealism, no gradients, no tiny text.

[continuous-batching-vs-static.png]
Flat vector timeline diagram, wide 16:9, dark graphite background. Top half "Static
batching": a long request and a short request start together; the slot for the short
request sits empty (coral hatching) after it finishes, waiting for the long one.
Bottom half "Continuous batching": as soon as the short request finishes, a new
request (amber) fills the slot immediately. Efficiency labels on right. Ivory text.

[speculative-decoding-accept-reject.png]
Flat vector pipeline diagram, wide 16:9, dark graphite background. Left: a small teal
"Draft" model proposes 5 tokens in sequence. Right: a large amber "Verifier" model
checks all 5 in one parallel forward pass. Tokens 1-3 are accepted (teal check), 4-5
are rejected (coral cross). The accepted 3 tokens are output. Time comparison shows
3x draft calls replaced 1 verifier call. Ivory labels.
```
