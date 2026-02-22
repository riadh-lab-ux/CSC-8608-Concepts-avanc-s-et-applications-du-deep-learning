import wave
import numpy as np
import torch

def rms(x: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(x ** 2)).item())

def clipping_rate(x: torch.Tensor, thr: float = 0.99) -> float:
    return float((x.abs() > thr).float().mean().item())

def load_wav_mono(path: str):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()  # bytes (2 for int16)
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth != 2:
        raise ValueError(f"Unsupported sampwidth={sampwidth} bytes. Expected 2 (16-bit PCM).")

    audio = np.frombuffer(raw, dtype=np.int16)  # little-endian on most wav pcm_s16le
    audio = audio.reshape(-1, n_channels)       # [time, channels]
    audio = audio.astype(np.float32) / 32768.0  # -> [-1, 1]

    # force mono
    audio_mono = audio.mean(axis=1, keepdims=True)  # [time, 1]
    wav = torch.from_numpy(audio_mono.T)            # [1, time]
    return wav, sr

def main():
    path = "TP3/data/call_01.wav"
    wav, sr = load_wav_mono(path)        # wav: [1, time]
    num_samples = wav.shape[1]
    duration_s = num_samples / sr

    print("path:", path)
    print("sr:", sr)
    print("shape:", tuple(wav.shape))
    print("duration_s:", round(duration_s, 2))
    print("rms:", round(rms(wav), 4))
    print("clipping_rate:", round(clipping_rate(wav), 4))

if __name__ == "__main__":
    main()

