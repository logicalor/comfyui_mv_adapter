# ComfyUI MV-Adapter Nodes

Custom nodes for [MV-Adapter](https://github.com/huanngzh/MV-Adapter) integration in ComfyUI. Generate multi-view consistent images from text or a single reference image.

## Features

- **Text-to-Multiview**: Generate multiple consistent views from text prompts
- **Image-to-Multiview**: Generate multiple views from a single reference image
- **Configurable camera angles**: Control azimuth/elevation for each view
- **Background removal**: BiRefNet (official) or rembg for reference images
- **SDXL support**: Works with SDXL model architecture
- **Bundled MV-Adapter code**: No separate MV-Adapter installation required
- **AMD/ROCm compatible**: Tested on AMD GPUs with automatic dtype detection

## Installation

1. Clone this repository into your ComfyUI `custom_nodes` folder:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/logicalor/comfyui-mvadapter.git
   ```

2. Install dependencies:
   ```bash
   cd comfyui-mvadapter
   pip install -r requirements.txt
   ```
   
   **Note for ROCm/AMD users**: The requirements.txt intentionally excludes `torch` and `torchvision` to avoid overwriting your ROCm-configured PyTorch installation. Make sure you have PyTorch properly configured for your hardware before installing.
   
   **Memory optimization for AMD/HIP**: If you encounter out-of-memory errors during VAE decode, try setting this environment variable to reduce memory fragmentation:
   ```bash
   export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
   ```

3. The adapter weights will be automatically downloaded from [HuggingFace](https://huggingface.co/huanngzh/mv-adapter) on first use.
   
   Alternatively, download manually and place in `ComfyUI/models/mvadapter/`:
   - `mvadapter_i2mv_sdxl.safetensors` for Image-to-Multiview (SDXL)
   - `mvadapter_t2mv_sdxl.safetensors` for Text-to-Multiview (SDXL)

---

## Quick Start: Image-to-Multiview

The simplest workflow to generate 6 consistent views from a single image:

```
Load Image → Background Removal → Reference Preprocess → I2MV Sampler → Image Grid
                                                              ↑
Pipeline Loader → Model Setup ─────────────────────────────────┤
                                                              ↑
Camera Embed ──────────────────────────────────────────────────┘
```

---

## Complete Usage Guide

### Step 1: Load the Pipeline

Use **MV-Adapter Pipeline Loader** to load the base SDXL model.

| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| `model_path` | `stabilityai/stable-diffusion-xl-base-1.0` | Downloads from HuggingFace automatically |
| `model_type` | `SDXL` | Only SDXL is fully supported |
| `auto_download` | `True` | Enable to download models automatically |
| `vae_id` | `madebyollin/sdxl-vae-fp16-fix` | **Recommended** - prevents NaN issues |
| `vae_name` | `none` | Use vae_id instead for best results |

**Outputs:**
- `pipeline` → Connect to Model Setup
- `vae` → Connect to VAE Decode (if using latents output)

### Step 2: Setup MV-Adapter

Use **MV-Adapter Model Setup** to initialize the adapter.

| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| `adapter_path` | `huanngzh/mv-adapter` | Official adapter weights |
| `adapter_type` | `i2mv` | For image-to-multiview (use `t2mv` for text-only) |
| `num_views` | `6` | Number of views to generate |

**Outputs:**
- `pipeline` → Connect to Scheduler Config or Sampler

### Step 3: Configure Scheduler (Optional but Recommended)

Use **MV-Adapter Scheduler Config** for improved quality.

| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| `scheduler_type` | `Euler` | Works well for most cases |
| `shift_snr` | `True` | **Important** - enables SNR shifting for better multi-view quality |
| `shift_scale` | `8.0` | Default from official implementation |

### Step 4: Setup Camera Embeddings

Use **MV-Adapter Camera Embed** to define view angles.

| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| `preset` | `orbit_6` | 6 views around object (0°, 45°, 90°, 180°, 270°, 315°) |
| `height` | `768` | Output height |
| `width` | `768` | Output width |
| `distance` | `1.8` | Camera distance |
| `elevation` | `0` | Camera elevation angle |

**Available presets:**
- `orbit_6` - 6 views around the object (recommended)
- `orbit_4` - 4 views (front, right, back, left)
- `front_back` - Just front and back views
- `custom` - Define your own azimuth angles

### Step 5: Prepare Reference Image

#### 5a. Background Removal (Recommended)

Use **MV-Adapter Background Removal** to isolate the subject.

| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| `method` | `birefnet` | Higher quality, matches official demo |
| `bg_color` | `gray` | Gray background works best |

**Alternative:** `rembg` is faster but lower quality

#### 5b. Reference Preprocessing (Recommended)

Use **MV-Adapter Reference Preprocess (Official)** to match the official demo's preprocessing.

| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| `target_size` | `768` | Match your output resolution |
| `object_scale` | `0.9` | Object fills 90% of frame |
| `bg_color` | `0.5` | Gray background (0.5 = 50% gray) |

This node:
- Crops to the object's bounding box
- Resizes to fill 90% of the frame
- Centers on a neutral background

### Step 6: Generate Views

Use **MV-Adapter I2MV Sampler** to generate the multi-view images.

| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| `prompt` | `high quality, best quality` | Keep simple for I2MV |
| `negative_prompt` | `watermark, ugly, deformed, noisy, blurry, low contrast` | Official default |
| `steps` | `30-50` | 30 for speed, 50 for quality |
| `guidance_scale` | `3.0` | Official default |
| `seed` | Any integer | For reproducibility |
| `reference_conditioning_scale` | `1.0` | How strongly to follow reference |
| `output_type` | `images` | Use `latents` for memory optimization |

**Outputs:**
- `images` → Your generated views (6 images in batch)
- `latents` → Raw latents (if output_type is `latents`)

### Step 7: View Results

#### Option A: Image Grid
Use **MV-Adapter Image Grid** to combine views:
- `columns`: `3` for a 3x2 grid of 6 views
- `padding`: `0` or small value

#### Option B: Split Views
Use **MV-Adapter Split Views** to separate into individual images.

---

## Memory Optimization (For Limited VRAM)

If you're running low on VRAM (under 16GB), use this optimized workflow:

### Latents + Separate VAE Decode

1. Set `output_type` to `latents` in the sampler
2. Add **MV-Adapter Clear VRAM** node after sampler
3. Add **MV-Adapter VAE Decode** to decode latents

```
Sampler (latents output) → Clear VRAM → VAE Decode → Images
```

The Clear VRAM node:
- Offloads the pipeline to CPU
- Clears GPU cache
- Allows VAE decode to use all available VRAM

### Additional Tips

- Enable `low_vram_mode` in the sampler for aggressive memory management
- Set `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` for AMD GPUs
- Use `batch_size: 1` in VAE Decode for minimum memory usage

---

## Text-to-Multiview (T2MV)

For generating views from text only (no reference image):

1. Use `adapter_type: t2mv` in Model Setup
2. Use **MV-Adapter T2MV Sampler** instead of I2MV
3. Write a detailed prompt describing the object

| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| `prompt` | Detailed description | Be specific about the object |
| `guidance_scale` | `7.0` | Higher than I2MV for text guidance |

---

## Complete Node Reference

### Pipeline & Model
| Node | Purpose |
|------|---------|
| **MV-Adapter Pipeline Loader** | Load base SDXL model |
| **MV-Adapter Model Setup** | Initialize MV-Adapter |
| **MV-Adapter Scheduler Config** | Configure scheduler with SNR shift |
| **MV-Adapter LoRA Loader** | Load optional LoRA weights |

### Camera & Views
| Node | Purpose |
|------|---------|
| **MV-Adapter Camera Embed** | Generate camera embeddings |
| **MV-Adapter View Selector** | Select specific views from output |
| **MV-Adapter Split Views** | Split batch into individual images |

### Image Processing
| Node | Purpose |
|------|---------|
| **MV-Adapter Background Removal** | Remove background (BiRefNet/rembg) |
| **MV-Adapter Reference Preprocess (Official)** | Official demo preprocessing |
| **MV-Adapter Image Preprocess** | Basic resize/pad preprocessing |
| **MV-Adapter Image Grid** | Combine images into grid |

### Sampling
| Node | Purpose |
|------|---------|
| **MV-Adapter I2MV Sampler** | Image-to-multiview generation |
| **MV-Adapter T2MV Sampler** | Text-to-multiview generation |

### Memory Management
| Node | Purpose |
|------|---------|
| **MV-Adapter Clear VRAM** | Free GPU memory between operations |
| **MV-Adapter VAE Decode** | Decode latents with memory optimization |

---

## Troubleshooting

### Black or NaN Images
- Use `madebyollin/sdxl-vae-fp16-fix` as the VAE
- The plugin auto-detects dtype, but float16 VAEs can cause issues

### Out of Memory
- Use latents output + Clear VRAM + VAE Decode workflow
- Enable `low_vram_mode` in sampler
- For AMD: `export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`

### Washed Out / Low Quality
- Enable `shift_snr: True` in Scheduler Config
- Use Background Removal + Reference Preprocess
- Try `reference_conditioning_scale: 1.0`

### Views Not Consistent
- Ensure you're using the same seed
- Check that camera embeddings match num_views in Model Setup
- Use `shift_snr: True` for better consistency

---

## Credits

- [MV-Adapter](https://github.com/huanngzh/MV-Adapter) by huanngzh
- [Official HuggingFace Demo](https://huggingface.co/spaces/VAST-AI/MV-Adapter-I2MV-SDXL)
- Based on research: "MV-Adapter: Multi-view Consistent Image Generation Made Easy"

## License

Apache-2.0
