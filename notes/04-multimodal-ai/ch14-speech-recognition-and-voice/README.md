# Chapter 14 · Speech Recognition and Voice Pipelines

> **Prerequisite chapters**: Ch.11 Audio Generation (TTS fundamentals, vocoder basics), Ch.10 Multimodal LLMs (encoder-decoder architecture, cross-modal representations)
>
> **What this chapter adds**: The upstream half of the voice loop. Ch.11 built text-to-speech synthesis — the output side. This chapter builds the input side: how raw microphone audio becomes text, how text becomes speech again, and how both halves are orchestrated under hard real-time constraints.

---

## What You'll Learn

By the end of this chapter you will be able to:

- Explain how a mel spectrogram captures speech information and why the mel scale is perceptually motivated rather than arbitrary
- Describe Whisper's encoder-decoder architecture and explain why 680K hours of noisy training data is its primary source of robustness
- Choose between batch and streaming ASR based on a given latency budget and explain the accuracy/latency tradeoff
- Trace a full voice assistant pipeline from wake word through TTS playback and account for every millisecond of a 1.5-second response budget
- Identify and mitigate the six failure modes most likely to cause production incidents in a deployed voice system

---

## Running Scenario

An in-car voice assistant for a premium EV brand. Four constraints drive every design decision in this chapter:

| Constraint | Value | Implication |
|------------|-------|-------------|
| **Latency** | < 1.5 s end-to-end | Every component has a fixed millisecond budget |
| **Noise floor** | 65–80 dB (road + HVAC + music) | ASR must be trained on or adapted to noise |
| **Connectivity** | Offline in tunnels and rural areas | No cloud API; on-device ASR required |
| **Speakers** | Multiple drivers, mixed accents | Model must generalize; single-speaker tuning fails |

---

## Chapter Contents

| Section | Topic |
|---------|-------|
| [§ 0 · The Challenge](speech-recognition-and-voice.md#0--the-challenge) | Latency budget decomposition, offline requirement |
| [§ 1 · How Speech Recognition Works](speech-recognition-and-voice.md#1--how-speech-recognition-works) | Mel spectrograms, HMM → DNN → Whisper evolution |
| [§ 2 · Whisper Architecture](speech-recognition-and-voice.md#2--whisper-architecture) | Encoder-decoder on audio tokens, model size tradeoffs |
| [§ 3 · Streaming vs Batch ASR](speech-recognition-and-voice.md#3--streaming-vs-batch-asr) | VAD, chunked inference, production choice |
| [§ 4 · TTS Pipeline](speech-recognition-and-voice.md#4--tts-pipeline) | Text normalization → G2P → acoustic model → vocoder |
| [§ 5 · Voice Assistant Architecture](speech-recognition-and-voice.md#5--voice-assistant-architecture-end-to-end) | End-to-end orchestration and latency accounting |
| [§ 6 · Failure Modes](speech-recognition-and-voice.md#6--failure-modes-and-real-world-gotchas) | Noise, accents, code-switching, confidence calibration |

---

## Files in This Chapter

| File | Purpose |
|------|---------|
| [speech-recognition-and-voice.md](speech-recognition-and-voice.md) | Full chapter content |

---

## Connection to Prior and Later Chapters

**← Ch.11 Audio Generation**: That chapter covered text-to-speech output — the acoustics model, MMS TTS, and vocoder. This chapter completes the TTS story (§4) with text normalization, grapheme-to-phoneme conversion, and vocoder selection details that Ch.11 treated as a black box. It also adds the full upstream ASR pipeline that Ch.11 never touched.

**→ Ch.15 Multimodal Fusion** *(planned)*: Acoustic representations from the Whisper encoder (transformer hidden states over mel features) are a concrete example of a learned unimodal embedding that feeds into cross-modal attention. The encoder-decoder pattern here is the same structural motif as vision-language fusion.
