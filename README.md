# Real-Time Voice Enhancement in Noisy/High-Threat Environments

Ever tried to shout over a loud bang and just... vanish? Most noise-suppression systems have this exact problem — when something really loud happens (a gunshot, an explosion, a slammed door), they don't just cancel the noise, they cancel *you* along with it. That's the problem we're solving here.

## What's going on

We're using DeepFilterNet3 to handle regular background noise, which it's genuinely great at. But like most denoisers, it can't tell "loud noise" apart from "loud noise with your voice buried inside it" — so the moment things spike, it suppresses everything, your voice included.

Our fix: split the incoming audio into two lanes in real time — the voice-frequency range and everything else. Crush the "everything else" lane hard during a spike, and boost the voice lane to make up for it, so your speech comes through even while the noise around it is getting slammed down.

```
mic → DeepFilterNet3 (denoise) → Sidechain Limiter → Voice Enhancer → De-esser → Noise Gate → Compressor → AGC → speaker
```

All of this runs live, block by block, using well under 15ms of processing time per block.

## Setup

Install the regular dependencies first:

```bash
pip install -r requirements.txt
```

**Heads up: `deepfilter_stream` is NOT on PyPI.** You can't just `pip install` your way to a working DeepFilterNet3 setup. It's a wrapper around the DeepFilterNet3 ONNX model, and you'll need to set it up manually — head over to the [DeepFilterNet repo](https://github.com/Rikorose/DeepFilterNet) and follow their instructions to get the model and streaming wrapper working before you try running this.

## Running it

```bash
python anc_pipeline.py
```

Use headphones — otherwise the speaker output will feed right back into the mic. Ctrl+C when you're done.

## What each stage does

- **DeepFilterNet3** — handles the heavy lifting on background noise
- **Sidechain Limiter** — the core fix: ducks non-voice content hard during loud spikes, boosts voice to compensate
- **Voice Enhancer** — EQ tuned to make speech sound clearer and more present
- **De-esser** — softens harsh "s" sounds that the EQ boost tends to introduce
- **Noise Gate** — mutes low-level hiss when nobody's talking
- **Compressor** — smooths out normal ups and downs in speaking volume
- **AGC** — keeps overall loudness consistent

## Tuning it

Every setting — thresholds, timing, EQ bands, gain caps — lives at the top of `anc_pipeline.py`. You shouldn't need to touch anything else for day-to-day adjustments.

## While it's running

You'll see two kinds of log lines:
- `[stats]` every few seconds — how much of your time budget processing is using, and whether any blocks got dropped
- `[event]` whenever something loud happens — shows what each stage did to the signal at that moment, useful for tuning

## What this isn't

This isn't true physical noise cancellation with inverse-phase audio — that needs hardware-level feedback loops running in sub-millisecond time, which a software mic-to-speaker pipeline just can't pull off. What we're doing is neural denoising plus smart real-time ducking, aimed specifically at the "can't hear myself during a loud transient" problem.

Latency floor sits around one processing block (~10.7ms) plus whatever the model itself takes to run — that's a hard limit since the model needs a full block of audio before it can even start.

## Team

Team name : AntiWave 

Team members: Ansh Agrawal , Nitesh Kumar , Raghav Gupta , Kritika Shukla , Jaanvi Jonwal , Riya Srivastava.





 
