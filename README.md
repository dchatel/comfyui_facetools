# comfyui_facetools

These custom nodes provide a rotation aware face extraction, paste back, and various face related masking options.

![Comparison](examples/comparison.jpg)

## Example Workflows
- Full inpainting workflow with two controlnets which allows to get as high as 1.0 denoise strength without messing things up.
![Full Inpainting](examples/full_inpainting.png)

## Nodes
- `AlignFaces`
  - inputs:
    - insightface: Use the `Load InsightFacenode` from [ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus)
    - image
    - threshold: minimal confidence score for detection
    - min_size: minimum face size for detection
    - max_size: maximum face size for detection
  - outputs:
    - faces
- `FaceDetails`
  - inputs:
    - faces
    - crop_size: size of the square cropped face image
    - crop_factor: enlarge the context around the face by this factor
    - mask_type:
      - simple_square: simple bounding box around the face
      - convex_hull: convex hull based on the face mesh obtained with MediaPipe
      - BiSeNet: occlusion aware face segmentation based on [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)
  - outputs:
    - crops: square cropped face images
    - masks: masks for each cropped face
    - warps: 2x3 warp matrix to paste the cropped face back into the original image
- `WarpFacesBack`
  - input:
    - images: this is the original image
    - face 
    - crop: cropped face images
    - mask: masks for the cropped faces
    - warp: 2x3 warp matrix
- `OrderedFaceFilter`
  - works similarly to `ImpactSEGSOrderedfilter`
- `GenderFaceFilter`
  - needs more work, as InsightFace gender classifier isn't very accurate.
- `MergeWarps`
  - I made some changes in `FaceDetailer` and `WarpFacesBack` since I've created this one, so it probably doesn't work anymore. Needs more testing and more work.

## Installation
You will need [ComfyUI-Impact-Pack](https://github.com/ltdrdata/) for `Load InsightFace` node and [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) for MediaPipe library (which is required for convex_hull masks) and `MediaPipe Face Mesh` node if you want to use that controlnet