# qwentts.cpp Build Notes

## Build Configuration

Built with ROCm/HIP GPU support on AMD Radeon Graphics (gfx1151, 122880 MiB VRAM).

### CMake Flags

```
CC=/usr/bin/cc CXX=/usr/bin/c++ HIPCXX=/opt/rocm-7.2.4/llvm/bin/clang++ cmake .. \
  -DCMAKE_C_COMPILER=/usr/bin/cc \
  -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm-7.2.4/llvm/bin/clang++ \
  -DCMAKE_HIP_PLATFORM=amd \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1151 \
  -DCMAKE_BUILD_TYPE=Release
```

### Build Artifacts

| Binary | Purpose |
|---|---|
| `tts-server` | OpenAI-compatible HTTP server for TTS (port 8081) |
| `qwen-tts` | CLI tool for TTS synthesis |
| `qwen-codec` | Audio codec CLI (encode WAV to latents) |
| `quantize` | GGUF requantizer |

### Model Files

Downloaded from [Serveurperso/Qwen3-TTS-GGUF](https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF):

| Model | Path | Size |
|---|---|---|
| qwen-talker-1.7b-customvoice-Q8_0.gguf | `models/qwen-talker-1.7b-customvoice-Q8_0.gguf` | ~1.9 GB |
| qwen-tokenizer-12hz-Q8_0.gguf | `models/qwen-tokenizer-12hz-Q8_0.gguf` | 278 MB |

The base model (`qwen-talker-1.7b-base-Q8_0.gguf`) has been replaced with the
`custom_voice` variant which supports speaker voice selection and voice cloning.

## Pre-registered Voices

The custom_voice model includes 9 pre-registered speakers:

| Voice | Dialect |
|---|---|
| `serena` | standard |
| `vivian` | standard |
| `uncle_fu` | standard |
| `ryan` | standard |
| `aiden` | standard |
| `ono_anna` | standard |
| `sohee` | standard |
| `eric` | sichuan_dialect |
| `dylan` | beijing_dialect |

Use the `voice` parameter to select one, e.g.:

```bash
curl -X POST http://localhost:8081/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-tts", "input": "Hello world", "voice": "serena", "response_format": "wav"}' \
  -o output.wav
```

New voices can be registered via POST `/v1/voices/<name>` with a reference audio file.

## Smoke Test Result

```bash
curl -X POST http://localhost:8081/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-tts", "input": "Hello, this is a test of the TTS system.", "response_format": "wav"}' \
  -o output.wav
```

Result: 128 KB valid WAV file, 24 kHz mono 16-bit PCM.
Performance: 1010.7 ms for 2.72 seconds of audio (RTF 0.372).

## Proxy Integration

The proxy expects `tts-server` at `localhost:8081` (config keys `tts_server_host`/`tts_server_port`).
Start the server with:

```bash
./tts-server \
  --model models/qwen-talker-1.7b-base-Q8_0.gguf \
  --codec models/qwen-tokenizer-12hz-Q8_0.gguf \
  --port 8081
```
