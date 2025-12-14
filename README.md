# ComfyUI MV-Adapter Nodes

Custom nodes for [MV-Adapter](https://github.com/huanngzh/MV-Adapter) integration in ComfyUI. Generate multi-view consistent images from text or a single reference image.

## Features

- **Text-to-Multiview**: Generate multiple consistent views from text prompts
- **Image-to-Multiview**: Generate multiple views from a single reference image
- **Configurable camera angles**: Control azimuth/elevation for each view
- **Background removal**: Built-in preprocessing for reference images
- **SDXL & SD2.1 support**: Works with both model architectures

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

3. Download MV-Adapter weights from [HuggingFace](https://huggingface.co/huanngzh/mv-adapter):
   - `mvadapter_i2mv_sdxl.safetensors` for Image-to-Multiview (SDXL)
   - `mvadapter_t2mv_sdxl.safetensors` for Text-to-Multiview (SDXL)
   
   Place them in `ComfyUI/models/mvadapter/`

## Nodes

### MVAdapterPipelineLoader
Loads the base diffusion model pipeline (SDXL or SD2.1).

### MVAdapterModelSetup
Initializes MV-Adapter attention processors and loads adapter weights.

### MVAdapterCameraEmbed
Generates camera embeddings (Plücker coordinates) for multi-view generation.

### MVAdapterI2MVSampler
Main sampling node for image-to-multiview generation.

### MVAdapterT2MVSampler
Main sampling node for text-to-multiview generation.

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
