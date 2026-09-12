import queue
import threading
import time

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
import sounddevice as sd
from deepfilter_stream import DeepFilterModel

SAMPLE_RATE = 48000
CHANNELS = 1
QUEUE_DEPTH = 2

ATTEN_LIM_DB = 12

LOW_CUT_HZ = 80
HIGH_CUT_HZ = 8000

BOOST_BANDS = [
    (500, 8, 1.0),
    (2000, 15, 0.7),
]

SIDECHAIN_THRESHOLD_DB = -14
SIDECHAIN_RATIO_DUCK_DB = 45
SIDECHAIN_ATTACK_MS = 0.3
SIDECHAIN_RELEASE_MS = 150
SIDECHAIN_VOICE_BOOST_MAX_DB = 12

AGC_TARGET_RMS_DB = -14
AGC_MAX_GAIN_DB = 30
AGC_TIME_CONSTANT_MS = 400

DEESS_LOW_HZ = 4000
DEESS_HIGH_HZ = 9000
DEESS_THRESHOLD_DB = -18
DEESS_ATTACK_MS = 0.5
DEESS_RELEASE_MS = 50

GATE_THRESHOLD_DB = -45
GATE_ATTACK_MS = 2
GATE_RELEASE_MS = 100
GATE_HOLD_MS = 50

COMP_THRESHOLD_DB = -20
COMP_RATIO = 3.0
COMP_ATTACK_MS = 5
COMP_RELEASE_MS = 100
COMP_MAKEUP_DB = 3

DIAG_LOG_PEAK_THRESHOLD = 0.05


def _peaking_eq_sos(f0, gain_db, q, fs):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    b0 = 1 + alpha * A
    b1 = -2 * cos_w0
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cos_w0
    a2 = 1 - alpha / A
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])


class VoiceBandEnhancer:
    def __init__(self, sr):
        self.sos_bandpass = butter(4, [LOW_CUT_HZ, HIGH_CUT_HZ], btype="bandpass", fs=sr, output="sos")
        self.zi_bp = sosfilt_zi(self.sos_bandpass)
        self.boost_sos = []
        self.boost_zi = []
        for freq, gain_db, q in BOOST_BANDS:
            sos = _peaking_eq_sos(freq, gain_db, q, sr)
            self.boost_sos.append(sos)
            self.boost_zi.append(sosfilt_zi(sos))

    def process(self, block):
        out, self.zi_bp = sosfilt(self.sos_bandpass, block, zi=self.zi_bp)
        for i in range(len(self.boost_sos)):
            out, self.boost_zi[i] = sosfilt(self.boost_sos[i], out, zi=self.boost_zi[i])
        return out.astype(np.float32)


class SidechainLimiter:
    def __init__(self, sr):
        self.sos_voice = butter(6, [LOW_CUT_HZ, HIGH_CUT_HZ], btype="bandpass", fs=sr, output="sos")
        self.zi_voice = sosfilt_zi(self.sos_voice)
        self.threshold = 10 ** (SIDECHAIN_THRESHOLD_DB / 20)
        self.duck_floor = 10 ** (-SIDECHAIN_RATIO_DUCK_DB / 20)
        self.attack_coef = np.exp(-1.0 / (sr * SIDECHAIN_ATTACK_MS / 1000))
        self.release_coef = np.exp(-1.0 / (sr * SIDECHAIN_RELEASE_MS / 1000))
        self.boost_max = 10 ** (SIDECHAIN_VOICE_BOOST_MAX_DB / 20) - 1.0
        self.envelope = 0.0
        self.gain = 1.0

    def process(self, block):
        voice, self.zi_voice = sosfilt(self.sos_voice, block, zi=self.zi_voice)
        residual = block - voice
        out_r = np.empty_like(residual)
        out_v = np.empty_like(voice)
        env, gain, thresh, atk, rel, floor, bmax = self.envelope, self.gain, self.threshold, self.attack_coef, self.release_coef, self.duck_floor, self.boost_max
        for i in range(len(block)):
            x = abs(block[i])
            env = atk*env+(1-atk)*x if x>env else rel*env+(1-rel)*x
            target = 1.0 if env<=thresh else max(floor, thresh/env)
            gain = atk*gain+(1-atk)*target if target<gain else rel*gain+(1-rel)*target
            out_r[i] = residual[i]*gain
            out_v[i] = voice[i]*(1.0+(1.0-gain)*bmax)
        self.envelope, self.gain = env, gain
        return (out_v+out_r).astype(np.float32)


class AutoGainControl:
    def __init__(self, sr, target_rms_db=AGC_TARGET_RMS_DB, max_gain_db=AGC_MAX_GAIN_DB, time_constant_ms=AGC_TIME_CONSTANT_MS):
        self.target_rms = 10 ** (target_rms_db / 20)
        self.max_gain = 10 ** (max_gain_db / 20)
        self.coef = np.exp(-1.0 / (sr * time_constant_ms / 1000))
        self.rms_est = 1e-6
        self.gain = 1.0

    def process(self, block):
        block_rms = float(np.sqrt(np.mean(block.astype(np.float64)**2))+1e-9)
        self.rms_est = self.coef*self.rms_est+(1-self.coef)*block_rms
        target = min(self.target_rms/self.rms_est, self.max_gain)
        target = max(target, 0.1)
        self.gain = self.coef*self.gain+(1-self.coef)*target
        return np.clip(block*self.gain, -1.0, 1.0).astype(np.float32)


class DeEsser:
    def __init__(self, sr):
        self.sos_band = butter(4, [DEESS_LOW_HZ, DEESS_HIGH_HZ], btype="bandpass", fs=sr, output="sos")
        self.zi_band = sosfilt_zi(self.sos_band)
        self.threshold = 10 ** (DEESS_THRESHOLD_DB / 20)
        self.attack_coef = np.exp(-1.0 / (sr * DEESS_ATTACK_MS / 1000))
        self.release_coef = np.exp(-1.0 / (sr * DEESS_RELEASE_MS / 1000))
        self.envelope = 0.0
        self.gain = 1.0

    def process(self, block):
        sibilant, self.zi_band = sosfilt(self.sos_band, block, zi=self.zi_band)
        complement = block - sibilant
        out_s = np.empty_like(sibilant)
        env, gain, thresh, atk, rel = self.envelope, self.gain, self.threshold, self.attack_coef, self.release_coef
        for i in range(len(sibilant)):
            x = abs(sibilant[i])
            env = atk*env+(1-atk)*x if x>env else rel*env+(1-rel)*x
            target = 1.0 if env<=thresh else thresh/env
            gain = atk*gain+(1-atk)*target if target<gain else rel*gain+(1-rel)*target
            out_s[i] = sibilant[i]*gain
        self.envelope, self.gain = env, gain
        return (complement+out_s).astype(np.float32)


class NoiseGate:
    def __init__(self, sr):
        self.threshold = 10 ** (GATE_THRESHOLD_DB / 20)
        self.attack_coef = np.exp(-1.0 / (sr * GATE_ATTACK_MS / 1000))
        self.release_coef = np.exp(-1.0 / (sr * GATE_RELEASE_MS / 1000))
        self.hold_samples = int(sr * GATE_HOLD_MS / 1000)
        self.envelope = 0.0
        self.gain = 0.0
        self.hold_counter = 0

    def process(self, block):
        out = np.empty_like(block)
        env, gain, hold, thresh, atk, rel = self.envelope, self.gain, self.hold_counter, self.threshold, self.attack_coef, self.release_coef
        for i in range(len(block)):
            level = abs(block[i])
            env = max(level, env*0.999)
            if env>thresh:
                target = 1.0
                hold = self.hold_samples
            elif hold>0:
                target = 1.0
                hold -= 1
            else:
                target = 0.0
            coef = atk if target>gain else rel
            gain = coef*gain+(1-coef)*target
            out[i] = block[i]*gain
        self.envelope, self.gain, self.hold_counter = env, gain, hold
        return out.astype(np.float32)


class Compressor:
    def __init__(self, sr):
        self.threshold = 10 ** (COMP_THRESHOLD_DB / 20)
        self.ratio = COMP_RATIO
        self.attack_coef = np.exp(-1.0 / (sr * COMP_ATTACK_MS / 1000))
        self.release_coef = np.exp(-1.0 / (sr * COMP_RELEASE_MS / 1000))
        self.makeup = 10 ** (COMP_MAKEUP_DB / 20)
        self.envelope = 0.0
        self.gain = 1.0

    def process(self, block):
        out = np.empty_like(block)
        env, gain, thresh, ratio, atk, rel = self.envelope, self.gain, self.threshold, self.ratio, self.attack_coef, self.release_coef
        for i in range(len(block)):
            level = abs(block[i])
            env = atk*env+(1-atk)*level if level>env else rel*env+(1-rel)*level
            if env>thresh:
                over_db = 20*np.log10(env/thresh+1e-9)
                reduced_db = over_db*(1-1/ratio)
                target = 10 ** (-reduced_db/20)
            else:
                target = 1.0
            gain = atk*gain+(1-atk)*target if target<gain else rel*gain+(1-rel)*target
            out[i] = block[i]*gain*self.makeup
        self.envelope, self.gain = env, gain
        return out.astype(np.float32)


print("Loading DeepFilterNet3 (ONNX)...")
model = DeepFilterModel()
df_stream = model.new_stream(atten_lim_db=ATTEN_LIM_DB)

BLOCK_SIZE = df_stream.frame_size
print(f"Using BLOCK_SIZE={BLOCK_SIZE} samples ({BLOCK_SIZE/SAMPLE_RATE*1000:.3f}ms)")

sidechain_limiter = SidechainLimiter(SAMPLE_RATE)
voice_enhancer = VoiceBandEnhancer(SAMPLE_RATE)
deesser = DeEsser(SAMPLE_RATE)
noise_gate = NoiseGate(SAMPLE_RATE)
compressor = Compressor(SAMPLE_RATE)
agc = AutoGainControl(SAMPLE_RATE)

_warmup_block = np.zeros(BLOCK_SIZE, dtype=np.float32)
for _ in range(3):
    df_stream.process_frame(_warmup_block)
print("Loaded.")

BLOCK_BUDGET_S = BLOCK_SIZE / SAMPLE_RATE
in_q = queue.Queue(maxsize=QUEUE_DEPTH)
out_q = queue.Queue(maxsize=QUEUE_DEPTH)
stop_event = threading.Event()

out_q.put(np.zeros(BLOCK_SIZE, dtype=np.float32))

proc_times = []
overrun_count = 0
callback_silence_fills = 0
input_drops = 0
output_drops = 0
stats_lock = threading.Lock()


def processing_loop():
    global overrun_count, output_drops
    while not stop_event.is_set():
        try:
            raw = in_q.get(timeout=0.5)
        except queue.Empty:
            continue
        t0 = time.perf_counter()
        rp = float(np.max(np.abs(raw)))
        d = np.asarray(df_stream.process_frame(raw), dtype=np.float32)
        d = np.pad(d, (0, BLOCK_SIZE-len(d))) if len(d)<BLOCK_SIZE else d[:BLOCK_SIZE]
        dp = float(np.max(np.abs(d)))
        l = sidechain_limiter.process(d); lg = sidechain_limiter.gain
        s = voice_enhancer.process(l)
        de = deesser.process(s)
        g = noise_gate.process(de); gg = noise_gate.gain
        c = compressor.process(g); cg = compressor.gain
        e = agc.process(c); ag = agc.gain
        op = float(np.max(np.abs(e)))
        if rp > DIAG_LOG_PEAK_THRESHOLD:
            print(f"[event] raw={rp:.3f}{' CLIP' if rp>=.999 else ''} dfn={dp:.3f} sc={lg:.3f} gate={gg:.3f} comp={cg:.3f} agc={ag:.3f} out={op:.3f}")
        el = time.perf_counter()-t0
        with stats_lock:
            proc_times.append(el)
            if el>BLOCK_BUDGET_S:
                overrun_count += 1
        try:
            out_q.put_nowait(e)
        except queue.Full:
            with stats_lock:
                output_drops += 1


def audio_callback(indata, outdata, frames, time_info, status):
    global callback_silence_fills, input_drops
    if status:
        print(status)
    try:
        in_q.put_nowait(indata[:, 0].copy())
    except queue.Full:
        with stats_lock:
            input_drops += 1
    try:
        processed = out_q.get_nowait()
        if len(processed) < frames:
            outdata[:, 0] = np.pad(processed, (0, frames-len(processed)))
        else:
            outdata[:, 0] = processed[:frames]
    except queue.Empty:
        outdata.fill(0)
        with stats_lock:
            callback_silence_fills += 1


def stats_reporter():
    while not stop_event.is_set():
        time.sleep(3)
        with stats_lock:
            if proc_times:
                avg = sum(proc_times)/len(proc_times)
                worst = max(proc_times)
                rtf = avg/BLOCK_BUDGET_S
                print(f"[stats] budget={BLOCK_BUDGET_S*1000:.1f}ms | avg={avg*1000:.1f}ms | worst={worst*1000:.1f}ms | RTF={rtf:.2f} | proc-overruns={overrun_count}/{len(proc_times)} | input-drops={input_drops} | output-drops={output_drops} | silence-fills={callback_silence_fills}")
                proc_times.clear()


def main():
    worker = threading.Thread(target=processing_loop, daemon=True)
    worker.start()
    reporter = threading.Thread(target=stats_reporter, daemon=True)
    reporter.start()
    try:
        with sd.Stream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=CHANNELS, dtype="float32", latency="low", callback=audio_callback):
            print(f"ACTIVE (atten_lim_db={ATTEN_LIM_DB}) — wear headphones — Ctrl+C to stop.")
            while not stop_event.is_set():
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop_event.set()
        worker.join(timeout=2)


if __name__ == "__main__":
    main()