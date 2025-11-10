# ComfyUI Step Audio EditX TTS

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Native-green.svg)](https://github.com/comfyanonymous/ComfyUI)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Step-Audio-EditX](https://img.shields.io/badge/Model-Step--Audio--EditX-orange.svg)](https://huggingface.co/stepfun-ai/Step-Audio-EditX)

Professional native ComfyUI nodes for **Step Audio EditX** - State-of-the-art zero-shot voice cloning and audio editing with emotion, style, speed control, and more.

## 🎯 Key Features

- **🎤 Voice Cloning**: Zero-shot voice cloning from 3-30 second reference audio
- **🎭 Audio Editing**: Edit emotion, speaking style, speed, add paralinguistic effects, and denoise
- **⚡ Native ComfyUI**: No JavaScript required - pure Python implementation
- **🧩 Modular Design**: Separate nodes for cloning and editing workflows
- **🎛️ Advanced Controls**: Full model configuration, generation parameters, and VRAM management
- **📊 Longform Support**: Smart chunking for unlimited text length in clone mode
- **🔄 Iterative Editing**: Multi-iteration editing for stronger effects

---

## 📦 Installation

<details>
<summary><b>Method 1: ComfyUI Manager (Recommended)</b></summary>

1. Open ComfyUI Manager
2. Search for "Step Audio EditX TTS"
3. Click Install
4. Restart ComfyUI

</details>

<details>
<summary><b>Method 2: Manual Installation</b></summary>

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/dannyson-sk/ComfyUI-Step_Audio_TTS.git
cd ComfyUI-Step_Audio_TTS
pip install -r requirements.txt
```

</details>

### 📥 Download Models

Download **both** the Step-Audio-EditX model and Step-Audio-Tokenizer. They must be in the correct folder structure:

```bash
cd ComfyUI/models
mkdir -p Step-Audio-EditX
cd Step-Audio-EditX

# Download BOTH repositories from HuggingFace
git clone https://huggingface.co/stepfun-ai/Step-Audio-EditX
git clone https://huggingface.co/stepfun-ai/Step-Audio-Tokenizer
```

**Required Folder Structure:**
```
ComfyUI/
└── models/
    └── Step-Audio-EditX/
        ├── Step-Audio-EditX/       ← Main model
        └── Step-Audio-Tokenizer/   ← Tokenizer (required)
```

Or download manually from:
- https://huggingface.co/stepfun-ai/Step-Audio-EditX
- https://huggingface.co/stepfun-ai/Step-Audio-Tokenizer

---

## 💻 VRAM Requirements

| Node | Precision | Quantization | VRAM Usage |
|------|-----------|--------------|------------|
| Clone | bfloat16 | none | ~11-14GB |
| Edit | bfloat16 | none | ~14-18GB |
| Clone | float16 | int8 | ~6-8GB |
| Edit | float16 | int8 | ~8-10GB |

**Note**: Edit node uses more VRAM than clone node. We are waiting for optimized quantized models from the Step AI research team and will implement them for lower VRAM usage when available.

**Recommendations**:
- **RTX 4090/A6000+**: Use bfloat16 + no quantization for best quality
- **RTX 3090/4080**: Use bfloat16 + int8 quantization
- **Lower VRAM**: Use int4 quantization (quality trade-off)

---

## 🎤 Clone Node

<details>
<summary><b>Overview</b></summary>

Zero-shot voice cloning from reference audio. The AI analyzes a 3-30 second voice sample and generates new speech in that voice with any text you provide.

**Use Cases**:
- Character voice generation
- Narration and voiceovers
- Voice consistency across long content
- Multilingual voice cloning

</details>

<details>
<summary><b>Parameters</b></summary>

### Text Inputs
- **prompt_text**: Exact transcript of reference audio (must match perfectly)
- **target_text**: New text to speak in cloned voice

### Model Configuration
- **model_path**: Path to Step-Audio-EditX model
- **device**: `cuda` (GPU, fast) or `cpu` (slow)
- **torch_dtype**: `bfloat16` (best), `float16` (good), `float32` (max quality), `auto`
- **quantization**: `none` (best), `int8` (good), `int4` (lower VRAM), `int4_awq`
- **attention_mechanism**: `sdpa` (default), `eager`, `flash_attn`, `sage_attn`

### Generation Parameters
- **temperature**: Voice variation (0.6-0.8 recommended for natural speech)
- **do_sample**: Keep `True` for natural voices
- **max_new_tokens**: Audio tokens to generate (4096 ≈ 20s, 8192 ≈ 40s)
- **longform_chunking**: Enable for text >2000 words (auto-splits and stitches)

### Advanced
- **seed**: 0 for random, or fixed number for reproducibility
- **keep_model_in_vram**: Keep loaded for speed or unload to free ~8GB VRAM

### Optional Input
- **prompt_audio**: Reference voice audio (3-30s recommended, 0.5-30s supported)

</details>

<details>
<summary><b>Example Settings</b></summary>

**High Quality, Long Content**:
```
temperature: 0.7
do_sample: True
max_new_tokens: 8192
longform_chunking: True
torch_dtype: bfloat16
quantization: none
```

**Fast, Lower VRAM**:
```
temperature: 0.7
do_sample: True
max_new_tokens: 4096
longform_chunking: False
torch_dtype: float16
quantization: int8
```

**Consistent Results**:
```
temperature: 0.5
do_sample: True
seed: 42
max_new_tokens: 4096
```

</details>

---

## 🎭 Edit Node

<details>
<summary><b>Overview</b></summary>

Edit existing audio with emotion, style, speed, paralinguistic effects, or denoising while preserving voice identity.

**Edit Types**:
- **Emotion**: happy, sad, angry, excited, calm, fearful, surprised, disgusted
- **Style**: whisper, gentle, serious, casual, formal, friendly
- **Speed**: faster (1.2x), slower (0.8x), more faster (1.5x), more slower (0.6x)
- **Paralinguistic**: [Laughter], [Breathing], [Sigh], [Gasp], [Cough]
- **Denoising**: denoise (remove noise), vad (remove silence)

</details>

<details>
<summary><b>Parameters</b></summary>

### Text & Audio Inputs
- **audio_text**: Exact transcript of input audio
- **input_audio**: Audio to edit (0.5-30 seconds)

### Model Configuration
Same as Clone Node (model_path, device, torch_dtype, quantization, attention_mechanism)

### Edit Configuration
- **edit_type**: Select category (emotion/style/speed/paralinguistic/denoising)
- **emotion**: Target emotion (only if edit_type=emotion)
- **style**: Speaking style (only if edit_type=style)
- **speed**: Speed adjustment (only if edit_type=speed)
- **paralinguistic**: Sound effect (only if edit_type=paralinguistic)
- **denoising**: Noise removal (only if edit_type=denoising)
- **paralinguistic_text**: Where to insert effect (leave empty to auto-append to end)
- **n_edit_iterations**: Edit strength (1=subtle, 2-3=moderate, 4-5=strong)

### Generation Parameters
- **temperature**: Hardcoded to 0.7 (parameter has no effect)
- **do_sample**: Hardcoded to True (parameter has no effect)
- **max_new_tokens**: Hardcoded to 8192 (parameter has no effect)

### Advanced
- **seed**: 0 for random, or fixed number for reproducibility
- **keep_model_in_vram**: Keep loaded or unload to free VRAM

</details>

<details>
<summary><b>Example Settings</b></summary>

**Add Emotion (Angry)**:
```
edit_type: emotion
emotion: angry
n_edit_iterations: 2
```

**Change Speaking Style (Whisper)**:
```
edit_type: style
style: whisper
n_edit_iterations: 3
```

**Speed Adjustment**:
```
edit_type: speed
speed: faster
n_edit_iterations: 1
```

**Add Laughter Effect**:
```
edit_type: paralinguistic
paralinguistic: [Laughter]
paralinguistic_text: (empty for auto-append to end)
n_edit_iterations: 1
```

**Clean Audio**:
```
edit_type: denoising
denoising: denoise
n_edit_iterations: 1
```

</details>

---

## 🎨 Workflow Examples

<details>
<summary><b>Basic Voice Cloning</b></summary>

1. Load reference audio (3-30s of voice sample)
2. Add **Step Audio Clone Node**
3. Connect reference audio to `prompt_audio`
4. Enter transcript in `prompt_text`
5. Enter new text in `target_text`
6. Set temperature to 0.7
7. Generate!

</details>

<details>
<summary><b>Long-Form Content with Chunking</b></summary>

1. Add **Step Audio Clone Node**
2. Enable `longform_chunking: True`
3. Set `max_new_tokens: 8192`
4. Enter long text (>2000 words) in `target_text`
5. Node will auto-split at sentence boundaries, generate chunks, and stitch seamlessly

</details>

<details>
<summary><b>Edit Audio Emotion</b></summary>

1. Load source audio to edit
2. Add **Step Audio Edit Node**
3. Connect audio to `input_audio`
4. Enter transcript in `audio_text`
5. Set `edit_type: emotion`
6. Choose emotion (e.g., `happy`)
7. Set `n_edit_iterations: 2` for moderate strength
8. Generate!

</details>

<details>
<summary><b>Clone + Edit Pipeline</b></summary>

1. Use **Clone Node** to generate speech in target voice
2. Connect clone output to **Edit Node** input
3. Apply emotion/style edits to the cloned voice
4. Fine-tune with multiple iterations

</details>

---

## ⚙️ Model Configuration Guide

<details>
<summary><b>Precision (torch_dtype)</b></summary>

- **bfloat16**: Best quality, stable, 11GB VRAM, recommended for RTX 40xx+
- **float16**: Good quality, 10GB VRAM, compatible with most GPUs
- **float32**: Maximum quality, 16GB VRAM, overkill for most use cases
- **auto**: Automatically selects best for your GPU

</details>

<details>
<summary><b>Quantization</b></summary>

- **none**: Best quality, highest VRAM (8GB+)
- **int8**: Good quality, medium VRAM (4-6GB), recommended for VRAM-constrained systems
- **int4**: Acceptable quality, low VRAM (3-4GB), noticeable quality loss
- **int4_awq**: Optimized int4, slightly better than standard int4

</details>

<details>
<summary><b>Attention Mechanism</b></summary>

- **sdpa**: Scaled Dot Product Attention (default, fastest, good VRAM)
- **eager**: Slowest but most stable, use for debugging
- **flash_attn**: Fastest, requires RTX 30xx+ with flash attention support
- **sage_attn**: Best VRAM efficiency, slightly slower than sdpa

</details>

---

## 🐛 Troubleshooting

<details>
<summary><b>Out of Memory (CUDA OOM)</b></summary>

1. Enable quantization (`int8` or `int4`)
2. Lower `max_new_tokens`
3. Disable `keep_model_in_vram` after each generation
4. Use `float16` instead of `bfloat16`
5. Close other GPU applications

</details>

<details>
<summary><b>Poor Voice Quality</b></summary>

1. Ensure `prompt_text` exactly matches reference audio
2. Use higher quality reference audio (clean, 3-30s)
3. Increase `temperature` for more natural variation
4. Disable quantization for best quality
5. Use `bfloat16` precision

</details>

<details>
<summary><b>Edit Not Strong Enough</b></summary>

1. Increase `n_edit_iterations` (try 3-4)
2. Ensure `audio_text` matches input audio exactly
3. Verify you selected correct edit category option (not "none")

</details>

<details>
<summary><b>Model Not Loading</b></summary>

1. Check model path: `ComfyUI/models/step_audio/Step-Audio-EditX`
2. Ensure all model files downloaded correctly
3. Verify sufficient disk space
4. Check ComfyUI console for detailed error messages

</details>

---

## 📚 Credits & License

**Model**: [Step-Audio-EditX](https://huggingface.co/stepfun-ai/Step-Audio-EditX) by StepFun AI
**ComfyUI Integration**: This custom node implementation
**License**: MIT

---

## 🤝 Contributing

Contributions welcome! Please open issues for bugs or feature requests.

---

## 📝 Changelog

<details>
<summary><b>Recent Updates</b></summary>

- ✅ Migrated to `max_new_tokens` for consistency across nodes
- ✅ Added paralinguistic auto-fill (auto-appends effect to end when text empty)
- ✅ Reordered edit node parameters for better UX
- ✅ Comprehensive tooltips for all parameters
- ✅ Fixed progress bar support (clone mode only)
- ✅ Enhanced VRAM management and caching

</details>

---

## 🔗 Links

- [Step Audio EditX Model](https://huggingface.co/stepfun-ai/Step-Audio-EditX)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Report Issues](https://github.com/dannyson-sk/ComfyUI-Step_Audio_TTS/issues)
