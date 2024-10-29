# toyxyz_tools
Easy-to-use AI tools. 

GUI with a collection of handy tools

![image](https://github.com/user-attachments/assets/067aba55-1ab1-4772-b909-9055cb2d7bb1)

This tool is intended for use on 2D illustrative images where normal estimation does not work well. 

Create a depth using Depthanything v2, and generate a normal map from the depth. 

ComfyUI version : https://github.com/toyxyz/ComfyUI_toyxyz_test_nodes

![image](https://github.com/user-attachments/assets/1b7f2ae4-f32a-4c06-882b-99c16bde8ccc)

## Example

Blender
![image](https://github.com/user-attachments/assets/d1e62c7f-408c-46ae-ae2f-3b6b22c7660f)

Clip studio paint
![image](https://github.com/user-attachments/assets/f1f92da0-f18f-43a4-b45c-643437c88684)

![image](https://github.com/user-attachments/assets/bdd04a01-1d7d-4499-968f-0c952d71fb55)


## Installation

  git clone https://github.com/toyxyz/toyxyz_tools

  cd toyxyz_tools

  python -m venv venv

  cd venv/Scripts

  activate

  pip install -r requirements.txt

  cd ../..

  gui.py

  !Install pytorch for the CUDA version installed on your machine. !

  https://pytorch.org/get-started/locally/

## How to use
  
  Add Image: Select one image. 

  Add Path: Select the path where the images are located. 

  D2N: Generates a plane mesh with a normal map from an image.  

  Lineart: Creates line art from an image. 

![image](https://github.com/user-attachments/assets/63bf5813-5911-44f3-951e-2640ba46371e)


## 1. D2N / Depth to normal 

  ![image](https://github.com/user-attachments/assets/7c8d95bd-28df-4e88-aa8a-484e20c760bf)
  
  Generates a depth using the input image, generates a normal map from the depth, and exports a .gITF 3D file. 

  If the input image has an alpha channel, it will also be applied to the glTF material. 

  You can import it into programs like Blender, Unity, or Clip studio paint. 

  ![image](https://github.com/user-attachments/assets/c09153ee-f805-4a68-a6bb-e6f165f6ef3d)

  Input_size: Specifies the resolution of the depth. Higher gives more detail, but uses more VRAM. 

  normal_depth: Adjusts the blue channel (depth) of the normal map: lower makes it look more three-dimensional, higher makes it flatter. 

  normal_min: Depths lower than this value are excluded from the normal map. 

  metallic: Specifies the metallic value of the material. 

  roughness: Specifies the roughness value of the material. 

  blur: Specifies the strength of the bilateral filter applied to the normal map. 

  sigmaColor :	Filter sigma in the color space.

  sigmaSpace :	Filter sigma in the coordinate space.

  Background : Specifies the color to fill in the background if the input image has an alpha channel. It's more effective to use a color that contrasts with the color of the object.  

  Upscale_tile : Specifies the tile size to use for normal map upscaling. If you lack vram, use a lower value. If it's 0, no tiles are used.

  Use color texture : Specifies whether to apply color texture to the mesh. White is used when disabled.

  Show BG Color: Displays the input image to be used for depth creation. Used to check background colors.

  Upscale normal: Upscales the normal map using the upscale model.

  Use path : Use images located in the path added as 'Add Path'.

  Save glTF: Saves the glTF file. If you only need normals and depth, turn this off. 

  Depth Model Selection: Select the DepthAnything v2 model to use to generate depth. Larger models use more VRAM and are more accurate. small < base < large

  It will automatically download the model. If you have any problems, download it from the path below and put it in toyxyz_tools/checkpoints. 

  https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true
  https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true
  https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true

  Upscale Model Selection: Select the RealESRGAN model to use for upscaling. 

  It will automatically download the model. If you have any problems, download it from the path below and put it in toyxyz_tools/weights. 

  https://github.com/xinntao/Real-ESRGAN

  Add Texture: Select an image to use as a color texture for the material instead of the input image. 

  Select Texture Directory: When using the 'Use Path' mode, select the path where the images to be used for the material's color texture are located. They should have the same name and format as the input images. 

  Select Output Directory: Specify the path where the generated image and glTF will be saved. If not selected, they will be saved in the output folder. 

  Generate : Generate start


## 2. Lieart extract

 ![image](https://github.com/user-attachments/assets/e9f199f1-a78c-431a-8fcf-ede51aed97d7)

  Generates line art from an input image. 

  ![image](https://github.com/user-attachments/assets/bd7818a2-72e9-4b34-8fe0-c0b4cb06a5ac)

  resolution: Specifies the resolution of the lineart. Higher values provide more detail, but use more resources. 

  blur_b, sigmaColor_b, sigmaSpace_b: Apply a bilateral filter to the image before it is processed. Increase the intensity if the image is noisy. 

  blur_a, sigmaColor_a, sigmaSpace_a: Apply a bilateral filter to the image after it is processed. Increase the intensity if the lineart image is noisy. 

  line_color: Specifies the color(RGB) of the line. White is not available. 

  Background : Specifies the color to fill in the background if the input image has an alpha channel. It's more effective to use a color that contrasts with the color of the object.  

  Upscale_tile : Specifies the tile size to use for lineart upscaling. If you lack vram, use a lower value. If it's 0, no tiles are used.

  Threshold: Sets pixels above the threshold to black and pixels below to white. Lowering it increases detail. 

  ![image](https://github.com/user-attachments/assets/1d032de5-4d4d-498d-a397-45907044ec05)


  Use Alpha: If the input image has an alpha channel, it will be applied to the line art image. 

  Show BG Color: Displays the input image to be used for depth creation. Used to check background colors.

  Upscale lineart: Upscales the lineart using the upscale model.

  Use path : Use images located in the path added as 'Add Path'.

  Use threshold: Use a threshold value. 

  Method selection: Specify the method to use for generating lineart. 

  Upscale Model Selection: Select the RealESRGAN model to use for upscaling. 

  Select Output Directory: Specify the path where the generated image and glTF will be saved. If not selected, they will be saved in the output folder. 

  Generate : Generate start


https://github.com/DepthAnything/Depth-Anything-V2

https://github.com/xinntao/Real-ESRGAN

