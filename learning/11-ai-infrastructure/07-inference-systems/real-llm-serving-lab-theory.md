# Serving a Real LLM: Handwritten Theory Notes

**Question:** why can a model fit in memory and answer one request, yet struggle
when the input gets longer or several people use it?

Use these notes with the [GPU serving lab](real-llm-serving-lab.ipynb).
It serves an unchanged public pretrained checkpoint, not a fine-tuned model.
The GPU experiments have not been executed on the authoring machine. Their
results must come from your own run, not from an expected-output table.

## 1. A working model is not yet a measured service

Generating text inside Python proves that a model can perform that calculation.
A server adds another boundary: a client sends a request, waits, receives text,
and eventually learns that the answer is finished.

Start with one real request. Pin the checkpoint, tokenizer revision, engine
version, precision, and request settings. Otherwise a later comparison might
change the model while appearing to change only the serving policy.

The lab uses a small model to make the experiment affordable. That model's
fluency is not the experiment, and its throughput is not a forecast for a large
deployment. A genuine small workload is useful evidence when its limits are clear.

**Complaint:** even one answer repeatedly uses information from its earlier tokens.

## 2. Avoid redoing work already completed

As an answer grows, the model needs information about the tokens already seen.
Recomputing all their useful attention representations at every step wastes work.
Keep those representations and append the new token's contribution instead.
This saved information is the **key/value cache**, usually shortened to KV cache.

The first comparison uses the same checkpoint and Transformers implementation
with caching disabled and enabled. Inputs, output length, and greedy decoding
remain fixed. Warm both paths before timing, alternate their order, and wait
for GPU work to finish before stopping the clock.

Check the generated token IDs as well as time. Different answers would not be a
clean comparison of an optimization that is supposed to preserve the calculation.
There is no required speedup: report what this workload actually did.

Do not compare this timing directly with the later vLLM timing. Changing engines
changes more than one variable, including kernels, scheduling, and HTTP overhead.

**Complaint:** the work we saved now occupies memory.

## 3. Model weights are only part of the memory bill

The weights do not grow when one prompt becomes longer. Saved token state does.
Several active requests also need cache space at the same time.

Picture three regions: model weights, saved token state, and runtime workspace.
“The model fits” accounts for only the first region. It does not establish that
the service can support the desired context length and number of users.

The lab counts bytes in actual key and value tensors after real forward passes.
That is tensor-storage evidence, not the whole process's GPU memory. The native
engine may reserve a cache pool ahead of time, round storage into blocks, and
need additional working memory.

Keep the distinction between **reserved capacity** and **currently occupied
capacity**. A fixed-size pool can remain allocated while requests occupy different
fractions of it. Flat process memory does not imply that cache pressure is absent.

**Complaint:** one request's limit and the service's shared capacity are different.

## 4. Separate the three limits

A request has an input and a requested output length. Their combined length must
fit the configured per-sequence context limit. The lab deliberately exceeds that
limit and requires the engine's context-length rejection, not just any HTTP error.

The cache pool is shared across active work. A request may fit individually while
several such requests cannot all retain their token states simultaneously.

The active-sequence limit controls scheduling. It does not automatically cap
every HTTP connection or the entire waiting queue. A service can therefore
accept more requests than it is currently executing.

Observe the outcome instead of predicting that the process must crash.
The engine may make requests wait or pause work and recompute it later. The
latter is **preemption**. A smaller pool need not produce a preemption in every run.

## 5. More users can improve throughput and worsen waiting

Several requests can share GPU execution work. That can use otherwise idle
capacity, but each active request also consumes memory and scheduling attention.
More concurrency is not automatically better.

Keep one tokenized workload and vary only the number of client workers.
Measure completed output tokens per second, request completion latency,
first-text latency, native running/waiting counts, and cache occupancy.

Client-observed first text is not an exact internal first-token timestamp.
A streamed chunk can contain several tokens. Likewise, a peak found by periodic
sampling is a sampled peak, not proof that nothing larger happened between samples.

A p95 from a dozen requests is a small-sample diagnostic, not a production promise.
Repeat the workload on a quiet GPU before making a capacity decision.

**Complaint:** later requests may repeat input work we already computed.

## 6. Reusing an opening is not reusing the whole answer

Within one generation, KV caching saves earlier token state. Across requests,
the engine may also reuse state for an identical input beginning. This is
**prefix caching**.

The lab starts a fresh engine with prefix caching enabled, submits the same
tokenized opening repeatedly, and reads native hit counters. It also checks
that each request still generates the requested output tokens.

The answer was not fetched from a whole-answer cache. Such a cache would skip
generation and return an earlier response; that belongs at a different layer,
covered in the gateway lessons.

Turning off vLLM prefix caching does not turn off KV reuse inside generation.
Confusing those switches would make the experiment answer the wrong question.

## 7. Keep the evidence, then compose the service

Retain the model revision, software versions, hardware, launch commands, inputs,
request settings, individual observations, and metric samples. Save failures
as failures; never replace an unavailable measurement with a plausible number.

The resulting evidence applies to one device, model, and workload. It does not
establish cloud behavior, production reliability, larger-model capacity, or an
improvement that multiplies with every other optimization.

Next, [Systems Composition](../09-systems-composition/README.md) asks who owns
admission, queue limits, deadlines, and scaling decisions around this real engine.

**Durable summary:** saving work consumes memory; sharing the GPU creates queues;
reuse and capacity must be measured together under one declared workload.
