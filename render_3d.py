import torch
from pygltflib import *
from pygltflib.utils import ImageFormat, Image
from PIL import Image as PILImage
import numpy as np
import base64
import io
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import tifffile
from cv2.ximgproc import guidedFilter

from depth_anything_v2.dpt import DepthAnythingV2

import upscale

def convert_to_gray(image, weights):
    
    normalized = image.astype(np.float32) / 255.0
    
    gray_image = np.dot(normalized[..., :3], weights)
    
    gray_image = np.clip(gray_image, 0.0, 1.0)
    
    gray_image = (gray_image * 255).astype(np.uint8)
    
    gray_cv2 = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    
    return gray_image
    
def multiply_image(image_a, image_b, blend_factor):
    
    img1 = image_a.astype(np.float32) / 255.0  # 0-1 범위로 정규화
    img2 = image_b.astype(np.float32) / 255.0
    
    blended_image = img1 * img2 # Multiply
    
    blended_image = img1 * (1 - blend_factor) + blended_image * blend_factor
    
    return blended_image


def render_depth_normal_mesh(input_img, input_size, out_dir, normal_depth, normal_min, mat_metallic, mat_roughness, normal_blur, blur_sigmacolor, blur_sigmaspace, depth_encoder, bg_color, enable_texture, show_preview, upscale_normal, upscale_model, save_mesh, use_path, tile_n, texture_path, detail_m, detail_b, detail_s, detail_c, sobel_ratio, guided_blur, guided_eps, guided_loop):


    # determine model paths

    model_path = os.path.join(f'checkpoints/depth_anything_v2_{depth_encoder}.pth')
    if not os.path.isfile(model_path):  # 모델 파일이 존재하는지 확인
        print(f"Downloading model to: {model_path}")
        from huggingface_hub import snapshot_download
        download_path = 'checkpoints'
        
        if depth_encoder == 'vits' :
            snapshot_download(repo_id="depth-anything/Depth-Anything-V2-Small", 
                                      allow_patterns=[f"*{depth_encoder}*"],
                                      local_dir=download_path, 
                                      local_dir_use_symlinks=False)
        if depth_encoder == 'vitb' :
            snapshot_download(repo_id="depth-anything/Depth-Anything-V2-Base", 
                                      allow_patterns=[f"*{depth_encoder}*"],
                                      local_dir=download_path, 
                                      local_dir_use_symlinks=False)
                                      
        if depth_encoder == 'vitl' :
            snapshot_download(repo_id="depth-anything/Depth-Anything-V2-Large", 
                                      allow_patterns=[f"*{depth_encoder}*"],
                                      local_dir=download_path, 
                                      local_dir_use_symlinks=False)
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_anything = DepthAnythingV2(**model_configs[depth_encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{depth_encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    # 이미지 경로
    image_path = input_img # 텍스처 이미지 경로

    blue_depth = normal_depth # 노멀맵 Blue 채널 깊이

    depth_min = normal_min # Depth 최소값

    Depth_size = input_size # Depth size

    metallic = mat_metallic # metallic 

    roughness = mat_roughness # roughness
    

    # 이미지 불러오기
    color_image = PILImage.open(image_path).convert("RGB")

    alpha_image = PILImage.open(image_path).convert("RGBA")
    
    gray_color = tuple(map(float, detail_c.split(",")))
    
    # 그레이스케일 이미지로 변환
    
    gray_image =  convert_to_gray(np.array(color_image), gray_color)
    
    blur_k = detail_b + (1-(detail_b%2))
    
    # 블러 추가
    gray_image = cv2.GaussianBlur(gray_image, (blur_k, blur_k), detail_s)
    
    # 배경 색상 선택
    background_color = tuple(map(int, bg_color.split(",")))
    
    # 배경 색상 배열 생성
    background_image = np.ones_like(np.array(color_image)) * background_color  
    
    # 알파 채널 추출
    alpha_channel = np.array(alpha_image)[:, :, 3]  # 알파 채널 가져오기 (0: R, 1: G, 2: B, 3: A)
    
    # [0, 1] 범위로 정규화
    alpha_mask = alpha_channel / 255.0
    
    alpha_mask_3 = cv2.cvtColor(alpha_mask.astype("uint8"), cv2.COLOR_GRAY2RGB)
    
    # 알파 마스크 적용: 배경 색상으로 채움
    color_image_filled = np.array(color_image) * alpha_mask[:, :, None] + background_image * (1 - alpha_mask[:, :, None])
    
    #color_image_filled =  np.array(color_image) * alpha_mask_3
    
    color_image_filled = cv2.cvtColor(color_image_filled.astype("uint8"), cv2.COLOR_RGB2BGR)
    
    depth_out = depth_anything.infer_image(color_image_filled, Depth_size)
            
    depth_out = (depth_out - depth_out.min()) / (depth_out.max() - depth_out.min()) * 255.0

    depth_array = depth_out
    
    if detail_m > 0 :
    
        depth_array = multiply_image(depth_array, gray_image, detail_m) #디테일 적용


    depth_array = depth_array * alpha_mask  # 알파 마스킹 적용
    

    depth_float32 = depth_array.astype(np.float32)
    
    depth_folder = os.path.join(out_dir, "depth")
    
    normal_folder = os.path.join(out_dir, "normal")
    
    os.makedirs(depth_folder, exist_ok=True)
    
    os.makedirs(normal_folder, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]  # 파일 이름에서 확장자 경로 제거 
    
    depth_base_name = os.path.join(depth_folder, base_name)
    
    normal_base_name = os.path.join(normal_folder, base_name)
    
    depth_image_path = f"{depth_base_name}.tiff"  # 저장할 깊이 이미지의 이름

    if depth_float32.max() > 1.0:  # Check if the input is in [0, 255] range
        print("Input detected in [0, 255] range. Normalizing to [0, 1].")
        depth_float32 = depth_float32 / 255.0
    tifffile.imwrite(depth_image_path, depth_float32, photometric='minisblack', metadata=None,)
    
    torch.cuda.empty_cache()



    def get_surface_normal_by_depth(depth, depth_m, mix_ratio, K=None):
        """
        depth: (h, w) of float, the unit of depth is meter
        K: (3, 3) of float, the depth camera's intrinsic
        """
        K = [[1, 0], [0, 1]] if K is None else K
        fx, fy = K[0][0], K[1][1]

        #depth_safe = np.where(depth == 0, np.finfo(np.float32).eps, depth)
        depth_safe = np.where(depth <= depth_m, np.finfo(np.float32).eps, depth)

        #dz_dv, dz_du = np.gradient(depth_safe)
        
        # np.gradient 계산
        dz_dv_grad, dz_du_grad = np.gradient(depth_safe)
        
        # sobel 계산
        dz_du_sobel = cv2.Sobel(depth_safe, cv2.CV_32F, 1, 0, ksize=1)
        dz_dv_sobel = cv2.Sobel(depth_safe, cv2.CV_32F, 0, 1, ksize=1)
        
        # 그래디언트 혼합
        dz_du = mix_ratio * dz_du_sobel + (1 - mix_ratio) * dz_du_grad
        dz_dv = mix_ratio * dz_dv_sobel + (1 - mix_ratio) * dz_dv_grad
        
        du_dx = fx / depth_safe
        dv_dy = fy / depth_safe

        dz_dx = dz_du * du_dx
        dz_dy = dz_dv * dv_dy

        normal_cross = np.dstack((-dz_dx, -dz_dy, np.ones_like(depth)))

        norm = np.linalg.norm(normal_cross, axis=2, keepdims=True)
        normal_unit = normal_cross / np.where(norm == 0, 1, norm)

        normal_unit[~np.isfinite(normal_unit).all(2)] = [0, 0, 1]
        
        return normal_unit
        
    depth = depth_float32

    if len(depth.shape) == 3:
        depth = depth[:, :, 0]

    K = np.array([[500, 0, 320],
                  [0, 500, 240],
                  [0, 0, 1]])

    vis_normal = lambda normal: np.uint8((normal + 1) / 2 * 255)[..., ::-1]

    # 법선 맵 계산
    normal1 = get_surface_normal_by_depth(depth, depth_min, sobel_ratio, K)
   
    if blur_sigmacolor <= 0:
        blur_sigmacolor = 1
    
    if blur_sigmaspace <= 0:
        blur_sigmaspace =1
        
    normal1_blurred = vis_normal(normal1)
    
    if guided_loop > 0 :
        
        for _ in range(guided_loop):
    
            if guided_blur > 0:
            
                normal1_blurred = cv2.ximgproc.guidedFilter(vis_normal(normal1), normal1_blurred, guided_blur, guided_eps)
    
    if normal_blur > 0:
    
        normal1_blurred = cv2.bilateralFilter(normal1_blurred, normal_blur, blur_sigmacolor, blur_sigmaspace)

    outputs = np.array(normal1_blurred).astype(np.float32) / 255.0
    outputs[..., 1] = 1.0 - outputs[..., 1] #Flip green channel

    blue_channel = outputs[..., 0]  
    blue_channel = blue_depth + blue_channel * (1.0 - blue_depth) # Remap blue channel
    outputs[..., 0] = blue_channel

    outputs= outputs * 255.0

    normal_image_path = f"{normal_base_name}.png"  # 저장할 노멀맵 이미지의 이름

    # 시각화 및 저장
    
    
    if upscale_normal :
        temp_image_path = f"{base_name}_temp.png" # 임시 노멀맵 저장
        cv2.imwrite(temp_image_path, outputs)
        outputs = upscale.upscale_image(temp_image_path, out_dir, upscale_model, tile_n, False, False)
        os.remove(temp_image_path)
    
    cv2.imwrite(normal_image_path, outputs)
    
    torch.cuda.empty_cache()
    
    #3D
    
    #if use_path is False :
    if texture_path is not None :
        image_path = str(texture_path)
    
    if save_mesh:
        
        color_tex = image_path # Get Color texture
        normal_tex = normal_image_path # Get normal texture
        depth_tex = depth_image_path # Get depth texture
        extra_color_text = str(texture_path)

        # GLTF 객체 생성
        gltf = GLTF2()
        scene = Scene()
        mesh = Mesh()
        primitive = Primitive()
        node = Node()
        buffer = Buffer()
        bufferView1 = BufferView()
        bufferView2 = BufferView()
        bufferView3 = BufferView()  # 텍스처 좌표용
        accessor1 = Accessor()
        accessor2 = Accessor()
        texcoord_accessor = Accessor()  # 텍스처 좌표 액세서
        texture = Texture()
        normal_texture = Texture()  # 노멀 맵을 위한 텍스처
        alpha_texture = Texture()  # 알파 텍스처를 위한 텍스처
        textureInfo = TextureInfo()
        normal_texture_info = TextureInfo()  # 노멀 맵 텍스처 정보
        alpha_texture_info = TextureInfo()  # 알파 텍스처 정보
        material = Material()
        material.pbrMetallicRoughness = PbrMetallicRoughness()  # 초기화
        sampler = Sampler()

        #이미지 크기 가져오기
        def get_image_size(image_path):
            with PILImage.open(image_path) as img:
                return img.size  # (width, height)

        # 이미지 파일을 Base64로 인코딩하는 함수
        def encode_image_to_base64(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
                
        def get_image_size_p(image_tex):
            
            return image_tex.size  # (width, height)

        # 이미지 파일을 Base64로 인코딩하는 함수
        def encode_image_to_base64_p(image_tex):
            # BytesIO 객체 생성
            buffered = io.BytesIO()
            # Pillow 이미지 객체를 PNG 형식으로 저장
            image_tex.save(buffered, format="BMP")
            # BytesIO에서 읽어서 Base64로 인코딩
            return base64.b64encode(buffered.getvalue()).decode('utf-8')

        # 텍스처 및 노멀 맵 이미지 크기 가져오기
        texture_size = get_image_size(color_tex)
        normal_map_size = get_image_size(normal_tex)

        # 평면의 정점 생성 (이미지 크기에 맞게 조정)
        width = texture_size[0] / 100.0  # 너비를 적절한 크기로 조정 (비율을 맞추기 위해 나누기)
        height = texture_size[1] / 100.0  # 높이도 적절히 조정

        vertices = [
            -width / 2, -height / 2, 0.0,  # Bottom-left
            width / 2, -height / 2, 0.0,   # Bottom-right
            width / 2, height / 2, 0.0,    # Top-right
            -width / 2, height / 2, 0.0,   # Top-left
        ]

        # 텍스처 좌표 생성 (Y 값을 반전)
        texture_coords = [
            0.0, 1.0,  # Bottom-left
            1.0, 1.0,  # Bottom-right
            1.0, 0.0,  # Top-right
            0.0, 0.0   # Top-left
        ]

        # 인덱스 데이터 생성
        indices = [0, 1, 2, 0, 2, 3]  # 2 triangles for the square

        # GLTF 버퍼 데이터 생성
        vertex_data = np.array(vertices, dtype=np.float32).tobytes()
        index_data = np.array(indices, dtype=np.uint16).tobytes()
        texcoord_data = np.array(texture_coords, dtype=np.float32).tobytes()

        buffer_data = vertex_data + index_data + texcoord_data
        buffer.uri = "data:application/octet-stream;base64," + base64.b64encode(buffer_data).decode('utf-8')
        buffer.byteLength = len(buffer_data)

        # 버퍼 뷰 설정
        bufferView1.buffer = 0
        bufferView1.byteOffset = 0
        bufferView1.byteLength = len(vertex_data)
        bufferView1.target = ARRAY_BUFFER

        bufferView2.buffer = 0
        bufferView2.byteOffset = len(vertex_data)
        bufferView2.byteLength = len(index_data)
        bufferView2.target = ELEMENT_ARRAY_BUFFER

        bufferView3.buffer = 0
        bufferView3.byteOffset = len(vertex_data) + len(index_data)
        bufferView3.byteLength = len(texcoord_data)
        bufferView3.target = ARRAY_BUFFER

        # 인덱스 액세서 설정
        accessor1.bufferView = 1
        accessor1.byteOffset = 0
        accessor1.componentType = UNSIGNED_SHORT
        accessor1.count = 6  # 6 인덱스
        accessor1.type = SCALAR
        accessor1.max = [3]
        accessor1.min = [0]

        # 정점 액세서 설정
        accessor2.bufferView = 0
        accessor2.byteOffset = 0
        accessor2.componentType = FLOAT
        accessor2.count = 4  # 4 정점
        accessor2.type = VEC3
        accessor2.max = [width / 2, height / 2, 0.0]
        accessor2.min = [-width / 2, -height / 2, 0.0]

        # 텍스처 좌표 액세서 설정
        texcoord_accessor.bufferView = 2  # 텍스처 좌표를 위한 뷰
        texcoord_accessor.byteOffset = 0
        texcoord_accessor.componentType = FLOAT
        texcoord_accessor.count = 4  # 4 텍스처 좌표
        texcoord_accessor.type = VEC2
        texcoord_accessor.max = [1.0, 1.0]
        texcoord_accessor.min = [0.0, 0.0]

        # GLTF 이미지 설정
        image = Image()
        if enable_texture :
            image.uri = "data:image/png;base64," + encode_image_to_base64(color_tex) # 텍스쳐 불러오기
        else :
            # 원본 텍스처 이미지 열기
            original_image = PILImage.open(color_tex)
            width, height = original_image.size
            
            # 알파 채널이 있는지 확인
            if original_image.mode in ('RGBA', 'LA'):
                # RGBA 모드로 변환하여 알파 채널 유지
                if original_image.mode != 'RGBA':
                    original_image = original_image.convert('RGBA')
                
                # 새로운 흰색 이미지 생성 (알파 채널 포함)
                white_image = PILImage.new('RGBA', (width, height), color=(255, 255, 255, 0))
                
                # 원본 이미지의 알파 채널을 흰색 이미지에 적용
                r, g, b, a = original_image.split()
                white_image.putalpha(a)
            else:
                # 알파 채널이 없는 경우 불투명한 흰색으로
                white_image = PILImage.new('RGB', (width, height), color='white')
            
            # 이미지를 base64로 인코딩
            white_buffer = io.BytesIO()
            white_image.save(white_buffer, format='PNG')
            white_bytes = white_buffer.getvalue()
            
            # GLTF 이미지 설정 - 흰색 사용
            image.uri = "data:image/png;base64," + base64.b64encode(white_bytes).decode()
        image.name = "My Texture"

        # 노멀 맵 이미지 설정
        normal_image = Image()
        normal_image.uri = "data:image/png;base64," + encode_image_to_base64(normal_tex)
        normal_image.name = "My Normal Map"

        # 알파 텍스처 이미지 설정
        alpha_image = Image()
        alpha_image.uri = "data:image/png;base64," + encode_image_to_base64(color_tex)  # 동일한 이미지 경로 사용 (예시)
        alpha_image.name = "My Alpha Texture"

        # 텍스처 및 샘플러 설정
        gltf.images.append(image)
        gltf.images.append(normal_image)
        gltf.images.append(alpha_image)  # 알파 텍스처 추가

        sampler.magFilter = NEAREST
        sampler.minFilter = NEAREST
        gltf.samplers.append(sampler)

        texture.source = 0  # 첫 번째 이미지 사용 (텍스처)
        texture.sampler = 0  # 첫 번째 샘플러 사용
        gltf.textures.append(texture)

        # 노멀 맵 텍스처 설정
        normal_texture.source = 1  # 두 번째 이미지 사용 (노멀 맵)
        normal_texture.sampler = 0  # 동일한 샘플러 사용
        gltf.textures.append(normal_texture)

        # 알파 텍스처 설정
        alpha_texture.source = 2  # 세 번째 이미지 사용 (알파 텍스처)
        alpha_texture.sampler = 0  # 동일한 샘플러 사용
        gltf.textures.append(alpha_texture)

        # 텍스처 정보 설정
        textureInfo.index = 0
        textureInfo.texCoord = 0

        # 노멀 맵 텍스처 정보 설정
        normal_texture_info.index = 1  # 노멀 맵 텍스처의 인덱스
        normal_texture_info.texCoord = 0  # 노멀 맵도 동일한 텍스처 좌표 사용

        # 알파 텍스처 정보 설정
        alpha_texture_info.index = 2  # 알파 텍스처의 인덱스
        alpha_texture_info.texCoord = 0  # 알파 텍스처도 동일한 텍스처 좌표 사용

        # 재질 설정
        material.pbrMetallicRoughness.baseColorTexture = textureInfo
        material.normalTexture = normal_texture_info  # 노멀 맵 텍스처 추가
        material.alphaTexture = alpha_texture_info  # 알파 텍스처 추가

        material.pbrMetallicRoughness.metallicFactor = metallic  # 메탈릭 값 설정
        material.pbrMetallicRoughness.roughnessFactor = roughness  # 메탈릭 값 설정

        # 알파 채널을 사용하는 경우 알파 모드와 알파 값 설정
        material.alphaMode = "BLEND"  # 알파 채널을 사용할 때

        # GLTF에 재질 추가
        gltf.materials.append(material)

        # 속성 설정
        primitive.attributes.POSITION = 1
        primitive.attributes.TEXCOORD_0 = 2  # 텍스처 좌표 추가
        primitive.indices = 0
        primitive.material = 0
        node.mesh = 0
        scene.nodes = [0]

        # GLTF 구조 조립
        gltf.scenes.append(scene)
        gltf.meshes.append(mesh)
        gltf.meshes[0].primitives.append(primitive)
        gltf.nodes.append(node)
        gltf.buffers.append(buffer)
        gltf.bufferViews.append(bufferView1)
        gltf.bufferViews.append(bufferView2)
        gltf.bufferViews.append(bufferView3)  # 텍스처 좌표 뷰 추가
        gltf.accessors.append(accessor1)
        gltf.accessors.append(accessor2)
        gltf.accessors.append(texcoord_accessor)  # 텍스처 좌표 액세서 추가
        
        gltf_folder = os.path.join(out_dir, "glTF")
        
        os.makedirs(gltf_folder, exist_ok=True)
        
        gltf_base_name = os.path.join(gltf_folder, base_name)
        
        # GLTF 파일 저장
        gltf.save(f"{gltf_base_name}.gltf")
        
        print(f"Save {gltf_base_name}.gltf")
    
    if show_preview:
        # 뎁스 생성에 사용되는 이미지를 작은 윈도우로 표시
        Preview_display = cv2.resize(color_image_filled, (int(512*(width/height)), 512))
        cv2.imshow('Close to proceed', Preview_display)
        if use_path:
            cv2.waitKey(100)  # 윈도우를 업데이트하고 대기
        else :
            cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return outputs

    
