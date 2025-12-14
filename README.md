# ComfyUI MV-Adapter Nodes

Custom nodes for [MV-Adapter](https://github.com/huanngzh/MV-Adapter) integration in ComfyUI. Generate multi-view consistent images from text or a single reference image.

## Features

- **Text-to-Multiview**: Generate multiple consistent views from text prompts
- **Image-to-Multiview**: Generate multiple views from a single reference image
- **Configurable camera angles**: Control azimuth/elevation for each view
- **Background removal**: Built-in preprocessing for reference images
- **SDXL support**: Works with SDXL model architecture
- **Bundled MV-Adapter code**: No separate MV-Adapter installation required

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

3. The adapter weights will be automatically downloaded from [HuggingFace](https://huggingface.co/huanngzh/mv-adapter) on first use.
   
   Alternatively, download manually and place in `ComfyUI/models/mvadapter/`:
   - `mvadapter_i2mv_sdxl.safetensors` for Image-to-Multiview (SDXL)
   - `mvadapter_t2mv_sdxl.safetensors` for Text-to-Multiview (SDXL)

## Nodes

### MVAdapterPipelineLoader
Loads the base diffusion model pipeline (SDXL).

### MVAdapterModelSetup
Initializes MV-Adapter attention processors and loads adapter weights.

### MVAdapterCameraEmbed
Generates camera embeddings (Plücker coordinates) for multi-view generation.

### MVAdapterI2MVSampler
Main sampling node for image-to-multiview generation.

**Parameters:**
- `low_vram_mode` (optional): Enable memory optimizations for GPUs with limited VRAM. When enabled:
  - Enables sequential CPU offload (moves model components to CPU when not in use)
  - Enables VAE tiling and slicing
  - Uses maximum attention slicing
  - More aggressive memory cleanup between operations
  
  This may slow down generation but significantly reduces peak VRAM usage.

### MVAdapterT2MVSampler
Main sampling node for text-to-multiview generation.

**Parameters:**
- `low_vram_mode` (optional): Same memory optimizations as I2MVSampler above.

### MVAdapterBackgroundRemoval
Removes background from reference images using rembg.

### MVAdapterViewSelector
Configure which views to generate (front, back, sides, etc.).

### MVAdapterImageGrid
Combines multi-view outputs into a single grid image.

## Example Workflow

1. Load your base SDXL model with **MVAdapterPipelineLoader**
2. Configure adapter with **MVAdapterModelSetup**
3. Generate camera embeddings with **MVAdapterCameraEmbed**
4. (Optional) Preprocess input image with **MVAdapterBackgroundRemoval**
5. Generate views with **MVAdapterI2MVSampler**
6. (Optional) Combine into grid with **MVAdapterImageGrid**

## Credits

- [MV-Adapter](https://github.com/huanngzh/MV-Adapter) by huanngzh
- Based on research: "MV-Adapter: Multi-view Consistent Image Generation Made Easy"

## License

Apache-2.0
