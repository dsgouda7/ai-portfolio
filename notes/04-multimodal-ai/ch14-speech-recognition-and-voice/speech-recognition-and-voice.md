# Speech Recognition and Voice Pipelines

> **The story.** On April 27, 1952, Bell Laboratories demonstrated "Audrey" — a seven-foot rack of electronics that could recognize spoken digits zero through nine, but only from a single pre-enrolled speaker and only after a 350-millisecond pause between each word. The machine achieved 97% accuracy on digits but was useless for anything else; its inventor, K. H. Davis, noted in the project report that "connected speech recognition remains an unsolved problem." Nineteen years later, DARPA launched its Speech Understanding Research program with a five-year mandate to build a system that could understand 1,000 words of continuous speech. Carnegie Mellon's Harpy, delivered in 1976, met the target — 1,011-word vocabulary, 95% sentence accuracy — but required 80× real-time processing on a minicomputer the size of a refrigerator. Real-time was not even a stated goal.
>
> The refrigerator shrank. In 1990, Dragon Dictate shipped as a commercial product for $9,000; it still required deliberate pauses between words. By 2001 IBM's ViaVoice and Dragon NaturallySpeaking had eliminated the pauses using Hidden Markov Models, but Word Error Rate on telephone speech stubbornly sat at 15–20%. The breakthrough arrived not from a better HMM but from abandoning HMMs entirely: in 2012 IBM's deep LSTM acoustic model cut error rates by 20–30% relative in a single generation. Baidu's Deep Speech in 2015 went further, replacing the entire handcrafted feature pipeline with a single end-to-end recurrent network trained on 11,940 hours of audio. Then, in September 2022, OpenAI released Whisper: trained on 680,000 hours of multilingual audio scraped from the internet — much of it noisy, accented, and in non-standard recording conditions — and the model simply absorbed the noise rather than fighting it. Whisper.tiny, at 39 million parameters, runs in 40 milliseconds on an ARM processor and achieves 5% Word Error Rate on clean English. The refrigerator now fits in a dashboard.
>
> **Why this matters.** Voice is the most natural human interface. It requires zero visual attention — critical in a moving vehicle — and can express nuance that a touchscreen cannot. But voice is also the most hostile modality for machine learning: the same sentence spoken by two people in different environments produces completely different waveforms. Building a voice system that works in a noisy car, across multiple accents, without a network connection is not an incremental engineering task. It requires understanding the full signal processing stack, the architectural choices that traded accuracy for robustness, and the real-time orchestration that turns isolated model calls into a coherent sub-second experience. That is what this chapter covers.

---

## Common Misconceptions

### 1. "Speech recognition is solved — just call a cloud API."

**Why it's seductive:** Google and AWS speech APIs achieve < 3% WER on clean English, produce JSON in milliseconds, and require ten lines of code. For a demo or a consumer app with reliable connectivity, this is entirely reasonable.

**The truth:** Cloud ASR fails in exactly the scenarios where voice is most valuable. In a car tunnel, a rural highway, or an aircraft there is no signal. Cloud latency adds 100–400 ms of round-trip time that does not exist in your latency budget. The API terms prohibit storing audio — a hard block for any system that needs to log utterances for compliance or improvement. And cloud ASR is a single point of failure: when the API degrades (it does), your entire voice interface goes dark. On-device ASR via Whisper.tiny costs 40 ms per utterance, runs offline, and is fully auditable. The tradeoff is a 4–6% WER increase on accented speech, which is recoverable through fine-tuning.

### 2. "The model recognizes words the way humans hear words — one at a time, left to right."

**Why it's seductive:** Human speech perception feels sequential. You hear a word, understand it, hear the next word. Streaming ASR systems emit partial transcripts that appear to follow this pattern.

**The truth:** Modern ASR models like Whisper operate on the entire utterance at once. Whisper encodes a 30-second mel spectrogram into a fixed-length acoustic representation before the decoder produces a single token. The decoder is autoregressive — it generates tokens left to right — but it has full access to the encoded representation of the *entire* utterance while doing so. This is why Whisper can correctly transcribe "I live on the banks of the Seine" even though "banks" is ambiguous until "Seine" resolves the context. Humans have this ability too; it is called the "phonemic restoration effect." The architecture mirrors human perception more closely than the streaming illusion suggests.

### 3. "TTS is just the reverse of ASR — the same model running backwards."

**Why it's seductive:** ASR maps audio → text; TTS maps text → audio. The symmetry suggests a single reversible model.

**The truth:** ASR and TTS are architecturally unrelated. ASR is primarily a classification problem: given an acoustic sequence, find the most probable token sequence. TTS is a generation problem: given a token sequence, synthesize a continuous-valued waveform that sounds natural, with correct prosody, at the right speaking rate, in a specific voice identity. The output space of TTS (16,000 audio samples per second) is many orders of magnitude higher-dimensional than the output space of ASR (tokens from a 50,000-entry vocabulary). A WaveNet-style vocoder generates audio autoregressively one sample at a time; nothing in ASR has an analogue. Modern end-to-end TTS systems (VITS, StyleTTS2) are closer in spirit to image generation models than to speech recognition.

---

## 0 · The Challenge

The in-car voice assistant must handle: "Navigate to the nearest charging station, but avoid the highway — I'm low on range and want to take the scenic route through Napa."

That sentence takes a human 3.1 seconds to speak. The requirement is total response time under 1.5 seconds from end of speech to start of spoken reply. The budget is therefore tighter than the utterance duration — the system cannot afford to wait until the driver finishes speaking. It must use streaming detection to know when speech ends, start processing immediately, and begin synthesizing the reply before the driver has even lifted their foot off the accelerator.

Here is where the 1,500 ms go:

| Stage | Median latency | P95 latency | Budget |
|-------|---------------|-------------|--------|
| Wake word detection | 30 ms | 45 ms | 30 ms |
| Voice Activity Detection (end-of-speech) | 20 ms | 35 ms | 20 ms |
| ASR transcription (Whisper.tiny) | 200 ms | 280 ms | 200 ms |
| NLU intent classification | 50 ms | 80 ms | 50 ms |
| LLM inference (Phi-3-mini 3.8B, q4) | 800 ms | 1050 ms | 800 ms |
| TTS synthesis (VITS-small) | 180 ms | 220 ms | 200 ms |
| Audio playback buffer (first 200 ms audible) | 200 ms | 200 ms | 200 ms |
| **Total** | **1,480 ms** | **1,910 ms** | **1,500 ms** |

The P95 column reveals the problem: under load or on a slower ARM chip, the P95 path busts the budget by 410 ms. Every component must be independently tunable. If the LLM is unavailable — no cellular, offline model not cached — the system must degrade gracefully to the intent classifier (50 ms) and canned responses, dropping total latency to 750 ms.

The offline constraint rules out Whisper.large (1.5B parameters, 1.2 GB, 800 ms inference on ARM). Whisper.tiny (39M parameters, 75 MB, 40 ms on ARM) fits in the latency budget with room. The cost: Word Error Rate rises from 2.7% (large-v3 on clean English) to 5.1% (tiny on clean English) and from 5.2% to 9.3% on noisy in-car conditions. For navigation commands, a 9% WER is acceptable — "navigate to Napa" will survive one substitution error. For open-ended queries it is not, which is why the system routes those exclusively to the LLM tier with its 800 ms budget.

*"The latency budget is not a goal — it is a contract. Every millisecond assigned to one component is stolen from another."*

---

## 1 · How Speech Recognition Works

### From pressure waves to tokens

A microphone converts air pressure variations into voltage. An analog-to-digital converter samples that voltage at 16,000 Hz (16 kHz) — 16,000 signed integers per second, each representing instantaneous air pressure. Whisper.tiny receives these integers, converts them to a mel spectrogram, and outputs a sequence of text tokens. The pipeline has three stages:

```
Raw audio (16 kHz PCM)
       │
       ▼
Short-Time Fourier Transform (STFT)
  window: 25 ms, hop: 10 ms
  → complex spectrogram (201 frequency bins × T frames)
       │
       ▼
Mel filterbank (80 triangular filters, mel-spaced)
  → mel spectrogram (80 bins × T frames)
       │
       ▼
Log compression: log(max(mel, 1e-10))
  → log-mel spectrogram  ← this is what the model sees
       │
       ▼
Whisper encoder + decoder
  → text tokens → string
```

The STFT divides the audio into overlapping windows and computes the frequency content of each window via a Fourier transform. The result is a matrix of complex numbers: rows are frequency bins, columns are time frames. Taking the magnitude squared gives the power spectrogram.

### Why mel scale?

Human hearing is not linear in frequency. We can distinguish a 200 Hz tone from a 300 Hz tone (100 Hz difference) far more easily than we can distinguish a 5,000 Hz tone from a 5,100 Hz tone (same 100 Hz difference). The cochlea acts as a compressive frequency analyzer — logarithmic at high frequencies. The mel scale models this compression:

$$m = 2595 \cdot \log_{10}\!\left(1 + \frac{f}{700}\right)$$

where $f$ is frequency in Hz and $m$ is the corresponding mel value. At low frequencies the mel scale is nearly linear; above ~1 kHz it becomes logarithmic. Applying 80 triangular filters spaced linearly in mel space clusters more filters in the perceptually important low-frequency region and fewer filters in the high-frequency region where human discrimination is weak. This is not an arbitrary design decision — it is why models trained on mel spectrograms generalize better than models trained on raw FFT features.

In-car speech sits primarily below 4 kHz. Road noise and HVAC occupy 100–800 Hz. Whisper's 80 mel bins concentrate resolution exactly where it is needed.

### The evolution of ASR architectures

| Era | Architecture | Key moment | WER (telephone) |
|-----|-------------|-----------|----------------|
| 1970s–2000s | Hidden Markov Models (HMMs) | CMU Sphinx, 1986 | 15–20% |
| 2012 | Deep LSTM acoustic model | IBM at ICASSP 2012 | 11–14% |
| 2014 | CTC + RNN | Graves & Jaitly 2014 | 8–10% |
| 2015 | End-to-end DNN | Baidu Deep Speech, 11.9K hours | 6–8% |
| 2019 | Self-supervised transformer | Facebook wav2vec 2.0 | 4–6% |
| 2022 | Weakly supervised, massive scale | OpenAI Whisper, 680K hours | 2–5% |

The trend is not architectural cleverness — it is scale. Every major jump corresponds to more training data or more compute, not a fundamentally new architecture. Whisper uses a standard encoder-decoder transformer that existed in 2017. What changed was feeding it 680,000 hours of audio.

*"The architecture was not the innovation. The willingness to train on noisy internet data was."*

---

## 2 · Whisper Architecture

Whisper is a seq2seq transformer trained on (audio, text) pairs. The encoder receives a log-mel spectrogram; the decoder produces text tokens autoregressively. Nothing in the architecture is novel — the novelty is in the training data distribution and the multi-task training objective.

### Encoder

The encoder processes a 30-second log-mel spectrogram as a 2D image. Two 1D convolutional layers (kernel width 3, stride 1 and 2) downsample the time dimension before the sequence enters the transformer:

```
Log-mel spectrogram: 80 mel bins × 3000 time frames (30 s at 10 ms hop)
          │
          ▼
Conv1D(80→512, k=3, stride=1) + GELU
          │
          ▼
Conv1D(512→512, k=3, stride=2) + GELU
          │
     1500 time frames, 512 channels
          │
          ▼
Sinusoidal positional embeddings added
          │
          ▼
N × Transformer encoder layers (self-attention + FFN)
          │
          ▼
Acoustic representation: 1500 × 512
```

The convolutional front-end is borrowed from vision transformers — it learns local feature detectors over the frequency–time plane, analogous to edge detectors over image patches. The transformer layers then integrate long-range temporal dependencies across the 1500-frame window.

### Decoder

The decoder is a standard causal transformer. Before any transcription tokens are generated, special control tokens are prepended to condition the decoder's behavior:

```
[SOT] [LANGUAGE_CODE] [TASK] [TIMESTAMPS_FLAG] ...
```

`[TASK]` is either `[TRANSCRIBE]` (output in source language) or `[TRANSLATE]` (output in English regardless of input language). This is the entirety of Whisper's multilingual and translation capability — a single special token. The decoder attends to both the encoder output (cross-attention) and the previously generated tokens (masked self-attention), then produces the next token from a 51,865-entry vocabulary (the standard Whisper multilingual tokenizer).

For the in-car assistant, `[LANGUAGE_CODE]` is set to `[EN]` and timestamps are disabled, saving roughly 15 ms of decoder overhead.

### Model size tradeoffs

| Model | Parameters | VRAM / RAM | WER (clean) | WER (noisy) | ARM latency |
|-------|-----------|-----------|-------------|-------------|-------------|
| tiny | 39 M | 75 MB | 5.1% | 9.3% | 40 ms |
| base | 74 M | 145 MB | 4.2% | 7.8% | 75 ms |
| small | 244 M | 480 MB | 3.1% | 5.9% | 210 ms |
| medium | 769 M | 1.5 GB | 2.9% | 5.1% | 650 ms |
| large-v3 | 1550 M | 3.1 GB | 2.7% | 4.8% | 1400 ms |

The in-car system uses `tiny`. The 4.2% absolute WER gap between tiny and large-v3 on noisy speech is not negligible — for an utterance with 8 words, it means roughly one extra error every 1.3 sentences. The decision to accept this gap is not an engineering preference; it is a constraint imposed by the 200 ms ASR budget.

### Word Error Rate

WER is the standard ASR evaluation metric. It is defined as the minimum edit distance between the reference transcript and the hypothesis, normalized by reference length:

$$\text{WER} = \frac{S + D + I}{N}$$

where $S$ is substitutions (wrong word), $D$ is deletions (missed word), $I$ is insertions (extra word), and $N$ is total words in the reference. WER can exceed 100% if the model inserts many words not present in the reference.

For in-car navigation commands ("navigate to," "call," "play," "set temperature to"), the vocabulary is constrained and the semantic impact of a single substitution is high — "navigate to Napa" becoming "navigate to papa" is a failure. The mitigation is constrained decoding: bias the decoder toward the expected vocabulary using logit processors that boost the log-probabilities of known navigation tokens.

*"WER measures transcript similarity, not task success. A 5% WER on navigation commands means one in twenty commands fails — which is not a 95% success rate, it is a 5% failure rate in the driver's experience."*

---

## 3 · Streaming vs Batch ASR

### Batch ASR

Batch ASR waits until the speaker stops, sends the complete utterance to the model, and returns a single transcript. Whisper is fundamentally a batch model — it processes a 30-second window in one forward pass. The advantages are:

- **Accuracy**: the model sees the full acoustic context before committing to any token. Ambiguous early words are resolved by later context.
- **Simplicity**: one inference call, one result, no partial-transcript management.
- **Consistent latency**: inference time is determined by utterance duration, not by real-time throughput constraints.

The disadvantage is user experience: the user must finish speaking before transcription begins. For a 3-second utterance, this adds 3 seconds of dead time *before* the 200 ms ASR budget even starts. The 1.5-second total response budget is violated before the model runs a single forward pass.

### Streaming ASR

Streaming ASR transcribes audio in chunks — typically 300–500 ms windows — emitting a partial transcript after each chunk and updating it as more speech arrives. The tradeoff:

- **Perceived latency**: the system can start processing while the user is still speaking. First partial transcript appears ~300 ms into the utterance.
- **Accuracy cost**: early chunks have no future context. The word "read" in the partial transcript may flip to a different pronunciation interpretation when "the book" arrives 400 ms later.
- **Complexity**: the consumer must handle transcript revisions gracefully. If the display shows "navigate to na—" and then updates to "navigate to Napa Valley," the UI must not flicker.

For in-car ASR, streaming is used differently: the audio is *buffered* in full while streaming VAD runs in parallel to detect end-of-speech. When the VAD fires, the complete buffered audio is sent to Whisper.tiny as a batch job. This hybrid approach gets the latency benefit (VAD fires 20 ms after speech ends, not after a configurable silence threshold) without the accuracy cost of partial transcription.

### Voice Activity Detection

VAD is the gatekeeper for the entire pipeline. It determines when speech starts and ends, filters out noise bursts that would waste ASR capacity, and defines the boundaries of the audio buffer sent to Whisper.

Silero VAD is the practical choice: a 1 MB ONNX model that runs 1 ms inference per 30 ms audio chunk, with >95% accuracy on speech/silence detection across noise conditions. It outputs a confidence score in [0, 1]. The in-car system uses:

- **Speech onset**: two consecutive chunks with confidence > 0.5 → begin buffering
- **Speech offset**: three consecutive chunks with confidence < 0.35 → flush buffer to ASR queue
- **Maximum buffer**: 8 seconds → force-flush regardless of VAD state (prevents stuck-in-speech scenarios)

The 20 ms VAD latency in the budget is the time from true speech end to VAD offset detection: three chunks × 10 ms chunk hop = 30 ms, minus the detection latency of the model's own context window ≈ 20 ms net.

---

## 4 · TTS Pipeline

The TTS pipeline converts the LLM's text response into an audio waveform. Four stages run sequentially, and each stage's failure mode is distinct from the others.

### Stage 1 — Text normalization

Raw LLM output contains tokens that have no canonical pronunciation: "3.5km," "Dr. Smith," "9:30 AM," "Route 101," "$42.50." Text normalization expands these into their spoken form before any phoneme lookup occurs:

- `3.5 km` → `three point five kilometers`
- `Dr. Smith` → `Doctor Smith`
- `9:30 AM` → `nine thirty AM`
- `Route 101` → `Route one oh one` (not "Route one hundred and one" — driving context)
- `$42.50` → `forty-two dollars and fifty cents`

This is not a neural problem — it is a finite-state transducer problem. Rule-based normalization handles 95% of cases. The remaining 5% (ambiguous abbreviations, domain-specific tokens) are handled by a small LM-based classifier that selects between candidate expansions. For the in-car system, a 500-entry domain-specific normalization dictionary covers navigation, time, distance, and currency without any model inference.

### Stage 2 — Grapheme-to-phoneme (G2P)

English spelling is irregular. "Read," "lead," and "bass" each have two distinct pronunciations depending on context; "colonel" is pronounced nothing like its spelling. G2P maps the normalized text to a phoneme sequence that the acoustic model can process:

`"Navigate to Napa Valley"` → `/n æ v ɪ ɡ eɪ t t ə n æ p ə v æ l i/`

The IPA-like sequence is the input to the acoustic model. Most production G2P systems are hybrid: a pronunciation dictionary (CMUdict covers 130,000 English words) for known words, and a sequence-to-sequence neural model for out-of-vocabulary proper nouns ("Schiaparelli," "Yountville," "Zuckerberg"). The neural G2P adds 8 ms inference overhead but prevents the embarrassing fallback of pronouncing "Yountville" as "yount-vill."

### Stage 3 — Acoustic model

The acoustic model predicts a mel spectrogram from the phoneme sequence, embedding prosodic information (duration per phoneme, pitch contour, speaking rate) into the prediction:

**FastSpeech 2** (2021): non-autoregressive duration predictor + mel decoder. Generates mel frames in parallel rather than one-by-one. Fast (< 50 ms for 10 words) but requires explicit duration supervision from a teacher model during training. Voice quality is clean but can sound slightly robotic in expressive passages.

**VITS** (2021): Variational Inference with adversarial Training for end-to-end TTS. Combines acoustic modeling and vocoding into a single model trained with a GAN objective. Naturalness is higher than FastSpeech 2; the end-to-end training avoids the error cascading between acoustic model and vocoder stages. Inference is 180 ms for a 10-word response on ARM.

For the in-car system: VITS-small. The single-model architecture eliminates one failure boundary and fits the 200 ms TTS budget.

### Stage 4 — Vocoder

If the acoustic model is separate (FastSpeech 2 route), a vocoder converts the predicted mel spectrogram to a raw waveform. The evolution of vocoders mirrors the evolution of ASR:

| Year | Model | Method | Realtime factor (CPU) |
|------|-------|--------|----------------------|
| 2016 | WaveNet | Autoregressive dilated convolutions | 0.02× (50× slower than realtime) |
| 2018 | WaveGlow | Normalizing flow | 1.1× (barely realtime) |
| 2020 | HiFi-GAN | Generative adversarial network | 50× realtime |
| 2021 | UnivNet | Multi-resolution discriminator | 80× realtime |

HiFi-GAN at 50× realtime means a 1-second speech segment synthesizes in 20 ms on CPU. This is the practical choice for any embedded system. The quality gap between HiFi-GAN and WaveNet is audible in A/B listening tests but imperceptible to users receiving navigational instructions at 65 mph.

*"The vocoder is the last mile of TTS. Users do not hear 'model quality' — they hear whether the voice sounds like a robot or a person."*

---

## 5 · Voice Assistant Architecture End-to-End

The full pipeline is an event-driven state machine with strict timing guarantees at each stage boundary.

### Pipeline overview

```
Microphone (continuous 16 kHz audio stream)
       │
       ▼ always-on
Wake Word Detector (1 MFLOP, 30 ms)
  "Hey Meridian" detected → trigger VAD start
       │
       ▼ streaming
Voice Activity Detection — Silero VAD (1 MB, 1 ms/chunk)
  Buffer audio chunks. On speech offset → flush to ASR queue
       │
       ▼ batch
ASR — Whisper.tiny (39 M params, 40 ms inference)
  Audio buffer → text transcript
       │
       ▼
NLU Intent Classifier (50 ms)
  Classify: navigation / media / climate / open-ended
       │
       ├─── navigation/media/climate ──► Intent handler (10 ms)
       │                                 ↓ canned or structured response
       └─── open-ended ──────────────► LLM — Phi-3-mini q4 (800 ms)
                                        ↓ natural language response
       │
       ▼
Text Normalization + G2P (15 ms)
       │
       ▼
TTS — VITS-small (180 ms)
       │
       ▼
Audio buffer → Speaker (first frame plays at T+1480 ms)
```

### Wake word

The wake word detector runs continuously on a dedicated microcontroller thread at effectively zero CPU cost from the application processor's perspective. Its model is approximately 50 KB — small enough to run on a Cortex-M4 at 1 MFLOP without interrupting the main application processor.

The threshold decision is asymmetric: false negatives (missed wake words) frustrate users more acutely than false positives (accidental triggers), but false positives in a moving vehicle create safety hazards — the driver hears an unexpected voice and is startled. The production threshold is tuned for:

- **False negative rate**: < 10% (system responds to 9 out of 10 genuine invocations)
- **False positive rate**: < 1 per 10 hours of ambient in-car audio

"Alexander" and "Allah" both have phoneme overlap with "Alexa." For a system named "Meridian," the nearest collision is "Maria" — a post-filter phoneme distance check adds 2 ms to wake word processing and reduces collisions by 80%.

### Intent routing

Not every utterance needs LLM inference. The intent classifier covers the 40% of in-car commands that follow predictable patterns:

| Intent | Example | Handler |
|--------|---------|---------|
| navigation | "Navigate to nearest charging station" | Structured API call |
| media | "Play Fleetwood Mac" | Music service query |
| climate | "Set temperature to 72" | CAN bus command |
| open-ended | "Should I take the highway or the scenic route?" | LLM inference |

Routing open-ended queries to the LLM is the only path that uses the 800 ms LLM budget. Structured intents complete in 50–100 ms total (classifier + handler), bringing total response time to ~700 ms — well inside budget. The intent classifier is a fine-tuned DistilBERT (66 M parameters, 8-bit quantized, 50 ms inference).

### Latency budget accounting

| Component | Median | P95 | Optimization lever |
|-----------|--------|-----|-------------------|
| Wake word | 30 ms | 45 ms | Threshold adjustment |
| VAD (end-of-speech) | 20 ms | 35 ms | Chunk size reduction |
| ASR | 200 ms | 280 ms | Model quantization (int8) |
| Intent classification | 50 ms | 80 ms | Classifier pruning |
| LLM (open-ended only) | 800 ms | 1050 ms | KV-cache warming, early exit |
| TTS synthesis | 180 ms | 220 ms | Streaming synthesis |
| First audio frame | 200 ms | 200 ms | Fixed (buffer requirement) |
| **Total (structured)** | **~700 ms** | **~860 ms** | — |
| **Total (LLM path)** | **1480 ms** | **1910 ms** | KV-cache most impactful |

The P95 LLM path at 1,910 ms busts the 1,500 ms budget. The production mitigation is **streaming TTS**: begin synthesizing the first sentence fragment of the LLM's response as tokens arrive, rather than waiting for the complete response. This reduces perceived latency by 200–300 ms at the cost of synthesis quality on sentence boundaries.

### Graceful degradation

| Condition | Degraded behavior | Latency impact |
|-----------|-----------------|----------------|
| No cellular signal | Skip LLM; intent classifier + canned responses | −800 ms |
| ASR timeout (> 400 ms) | Return empty transcript; prompt re-speak | 0 |
| TTS synthesis error | Fall back to pre-recorded audio clips | −100 ms |
| VAD stuck in speech | Force-flush at 8-second maximum buffer | +variable |
| Wake word false positive | Cancel pipeline at VAD stage if < 300 ms audio | +0 ms |

Degradation is explicit and monitorable. Every fallback increments a Prometheus counter. If the no-LLM fallback fires more than 15% of sessions, the caching layer is warming the wrong model.

---

## 6 · Failure Modes and Real-World Gotchas

### 1. Background noise

Road noise at 70 mph produces a broadband noise floor at 65 dB SPL. HVAC at full power adds another 5 dB in the 200–800 Hz range — directly overlapping with vowel formants. A model trained only on clean speech sees this as acoustic garbage and either transcribes noise phonemes or produces hallucinated words ("the the the" is a classic Whisper hallucination in high-noise conditions).

**Mitigation**: Whisper.tiny's 680K-hour training corpus includes noisy internet audio. For noisy conditions, the `condition_on_previous_text=False` flag prevents the decoder from using a hallucinated prefix to generate additional hallucinations. Spectral subtraction preprocessing (estimate noise profile from the first 200 ms before speech onset; subtract it from all subsequent frames) reduces effective noise floor by 6–10 dB. In extreme cases, a multi-microphone beamformer — standard in modern EV interiors — provides 10–12 dB signal-to-noise improvement before ASR even runs.

### 2. Accents and dialects

Whisper.tiny achieves 5.1% WER on American English. On Indian English accents, WER rises to 11.3%. On rural Southern American English, 8.7%. The model is not broken — these accents appear less frequently in the 680K-hour training corpus.

**Mitigation**: Fine-tune Whisper.tiny on 5–10 hours of in-domain accented speech. This is achievable per regional market. Fine-tuning on Southern American English data reduces WER from 8.7% to 6.1% (a 30% relative improvement) with 8 hours of crowd-sourced recordings. The fine-tuned adapter is a 12 MB LoRA delta — small enough to deploy as a market-specific model variant without maintaining a separate full checkpoint per region.

### 3. Code-switching

Mid-sentence language switches are common in multilingual urban environments: "Navigate to die Autobahn" (German article mid-English sentence), "Call mi hermano" (Spanish noun in English command). English-only ASR models handle this by hallucinating English homophones: "die Autobahn" → "die out of bond."

**Mitigation**: Use the multilingual Whisper.tiny checkpoint rather than the English-only variant. The multilingual model adds 8% inference overhead (the language-detection head runs at the start of decoding) and 1.3% WER on pure English speech — a modest cost for correct handling of code-switched utterances. In markets where code-switching is common (Singapore, Miami, Berlin), the multilingual checkpoint is not optional.

### 4. Confidence calibration

Neural ASR models are systematically overconfident. A logit of 15.2 on the token "Napa" feels like a certainty, but the model has never seen "Napa" in a context with a broken microphone, and its confidence estimate does not account for distributional shift.

The mean log-probability of all generated tokens is the cheapest calibration signal:

$$\bar{\ell} = \frac{1}{T} \sum_{t=1}^{T} \log p(\hat{y}_t \mid \hat{y}_{< t}, X)$$

where $\hat{y}_t$ is the predicted token at step $t$ and $X$ is the encoder output. If $\bar{\ell} < -0.4$, the transcript is considered low-confidence.

**Production threshold behavior**:
- $\bar{\ell} \geq -0.2$: accept transcript, proceed
- $-0.4 \leq \bar{\ell} < -0.2$: accept transcript, flag for logging
- $\bar{\ell} < -0.4$: prompt re-speak ("I didn't catch that — could you repeat?")

The -0.4 threshold was calibrated on 2,000 held-out in-car utterances: it catches 78% of transcription errors while adding a re-speak prompt to only 6% of correct transcriptions — a false positive rate the UX team judged acceptable.

### 5. Hotword spotter false positives

"Alexander" triggers "Alexa." "Maria" partially triggers "Meridian." These are phonemic overlaps, not model bugs.

**Mitigation**: Post-filter with phoneme edit distance. After wake word detection fires, compute the Levenshtein distance between the detected phoneme sequence and the phoneme sequences of known collision words. If the detected sequence is closer to a collision word than to the target wake word by more than one phoneme edit, discard the trigger. This 2 ms post-filter reduces false positive rate by 80% on names containing the target phoneme cluster.

### 6. Whisper hallucination on silence

Whisper is trained on speech audio. When given silence or non-speech noise, it does not output an empty string — it outputs plausible-sounding but completely fabricated transcriptions. This is a known artifact of training on 680K hours of audio where near-silent segments were rare.

**Mitigation**: Always gate ASR with VAD. Never send audio to Whisper unless Silero VAD has confirmed speech is present. If the VAD buffer drains without a confident speech detection (confidence never exceeded 0.5 in 1 second), discard the buffer and do not call ASR. The additional 1 ms per chunk that Silero VAD costs is cheap insurance against Whisper hallucinating "thank you" into every moment of road silence.

*"The model does not know what it does not know. VAD is the part of the pipeline that knows the difference between silence and speech."*

---

## 7 · When to Use What — Decision Framework

### ASR model choice

| Scenario | Recommended model | Reason |
|----------|-----------------|--------|
| Cloud-connected consumer app, accuracy critical | Whisper large-v3 via API or self-hosted | Best accuracy, latency not primary constraint |
| On-device mobile, English only | Whisper.small (244 M, int8) | Accuracy/size balance; fits 500 MB RAM budget |
| On-device embedded / automotive | Whisper.tiny (39 M) | Fits 200 ms latency budget; 75 MB footprint |
| Real-time transcription, low latency | Faster-Whisper + streaming VAD | CTranslate2 backend, 4× faster than vanilla |
| Multilingual, code-switching | Whisper multilingual checkpoint | Language-detection head adds < 10% overhead |
| Custom domain (medical, legal) | Fine-tuned Whisper.base | Domain vocabulary; 10 h of labeled audio sufficient |

### TTS model choice

| Scenario | Recommended model | Reason |
|----------|-----------------|--------|
| Maximum naturalness, cloud OK | ElevenLabs / GPT-4o-mini-TTS | State-of-the-art prosody; requires API |
| On-device, single-speaker | VITS-small with speaker embedding | End-to-end, no vocoder seam; 180 ms on ARM |
| Multi-speaker, fast synthesis | FastSpeech2 + HiFi-GAN | Non-autoregressive; fastest CPU inference |
| Multilingual, 1,000+ languages | Meta MMS TTS | Breadth over quality; zero-shot language switching |
| Low-resource language | Meta MMS fine-tune | Pre-trained on 1,100 languages; 30 min fine-tuning |

### VAD choice

Silero VAD is the default for any production system: 1 MB, 1 ms per chunk, ONNX-exportable. The only alternative worth considering is WebRTC VAD — it is already bundled in many audio frameworks and adds zero dependency overhead, but its accuracy on noisy conditions is 5–8% worse than Silero. Use WebRTC VAD only in packaging-constrained environments where adding a 1 MB dependency is genuinely prohibited.

---

## 8 · Interview Checklist

**Core ASR concepts**

- Explain the mel spectrogram: why mel scale, what does it encode, how does it relate to the STFT?
- Why is end-to-end ASR (Whisper, Deep Speech) better than HMM-based ASR? What did HMMs assume that was wrong?
- Whisper is described as "robust." What specifically does that mean and why is it true?
- What is WER and what are its limitations as a production metric?
- What is the difference between batch ASR and streaming ASR? Under what constraints would you choose each?

**Voice assistant architecture**

- Draw the pipeline from microphone to speaker output. What are the five stages and their latency budgets?
- What is VAD and why is it necessary if Whisper already handles silence gracefully? (It does not — see §6.)
- How do you handle the tradeoff between wake word false negatives and false positives in a safety-critical context?
- What is graceful degradation and what triggers it in a production voice system?

**TTS**

- What is grapheme-to-phoneme conversion and why is a dictionary insufficient for all English words?
- Why did WaveNet autoregressive vocoders fail in production, and how did HiFi-GAN solve the problem?
- What is the difference between FastSpeech 2 and VITS architecturally, and when would you choose each?

**Failure modes**

- A deployed ASR system performs well on test data but users report frequent errors in production. What are five hypotheses and how would you investigate each?
- What is confidence calibration in ASR and how would you implement a re-speak prompt trigger?
- A Whisper deployment starts outputting "thank you, thank you, thank you" continuously when no one is speaking. What is the bug and how do you fix it?

---

## 9 · Progress Check — What Have We Unlocked?

### Before this chapter

The voice assistant built in `learning/ibm-genai/voice_assistant/` was a working system without a theoretical foundation. The TTS output was explained by Ch.11 (MMS TTS, vocoder basics). The ASR input was a black box — "use the Whisper API" — with no understanding of why Whisper is robust, what its failure modes are, or why a 39M-parameter model is the right choice for an automotive deployment.

The latency budget was unknown: the system either worked or did not, with no framework for diagnosing *where* it was slow.

### After this chapter

- **ASR** is not a black box. The mel spectrogram → encoder → decoder path is legible. Model size selection is a principled decision (WER vs latency on ARM) not a guess.
- **Latency** has a budget. Every millisecond is owned by a named component. When the system is slow, the table in §5 points to the responsible stage.
- **Failure modes** are enumerable. Background noise, accents, code-switching, hallucination on silence, hotword false positives — each has a specific mitigation that can be implemented independently.
- **TTS** is fully explained. Ch.11 left text normalization, G2P, and vocoder selection as details; §4 closes those gaps.
- **The IBM voice assistant** now has a theoretical dual. The Docker container is the implementation; this chapter is the mental model.

---

## Further Reading

| Resource | Why read it |
|----------|------------|
| Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision," OpenAI 2022 | Primary Whisper paper; §4 on training data curation explains why noisy data improved robustness |
| Silero VAD GitHub | Implementation details for the 1 MB VAD model; includes ONNX export for embedded deployment |
| Kong et al., "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis," NeurIPS 2020 | Vocoder paper; §3 on multi-period discriminators explains the quality improvement over WaveNet |
| Kim et al., "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech," ICML 2021 | VITS paper; end-to-end TTS removes the acoustic model / vocoder seam |
| Pratap et al., "Scaling Speech Technology to 1,000+ Languages," Meta AI 2023 | MMS paper (Ch.11 model); relevant for multilingual deployment across accents |

---

## Bridge to Chapter 15

The voice pipeline built in this chapter produces two high-dimensional representations at runtime: the Whisper encoder output (1500 × 512 acoustic features) and the LLM's hidden states over the text response (T × 4096). Both are learned embeddings of the same underlying intent — the driver's request — expressed in different modalities.

Chapter 15 examines what happens when a system learns to align those representations: multimodal fusion architectures that process audio and text jointly rather than sequentially. The encoder-decoder separation that made Whisper tractable becomes a design constraint to relax. The same cross-attention mechanism that lets Whisper's decoder query its encoder will be the primitive operation for audio-text fusion at inference time.
