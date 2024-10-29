from PIL import Image
import cv2
import numpy as np
import torch
from controlnet_aux import AnylineDetector, TEEDdetector, LineartDetector, LineartAnimeDetector
import upscale
import os
import math

def color_blend(base_image, color_image):
    # 이미지를 HSV 색상 공간으로 변환
    base_hsv = cv2.cvtColor(base_image, cv2.COLOR_RGB2HSV)
    color_hsv = cv2.cvtColor(255-color_image, cv2.COLOR_BGR2HSV)
    
    # 밝기(V) 채널은 base_image에서, 색상(H)과 채도(S) 채널은 color_image에서 가져옴
    blended_hsv = np.zeros_like(base_hsv)
    blended_hsv[:, :, 0] = color_hsv[:, :, 0]  # H 채널 (색상)
    blended_hsv[:, :, 1] = color_hsv[:, :, 1] # S 채널 (채도)
    blended_hsv[:, :, 2] = 255-base_hsv[:, :, 2]   # V 채널 (명도)

    # HSV에서 다시 BGR로 변환
    blended_image = cv2.cvtColor(blended_hsv, cv2.COLOR_HSV2BGR)
    
    return (255-blended_image)
    
def create_solid_color_image(image_np, rgb_color):
    # 이미지의 높이와 너비 가져오기
    height, width, channels = image_np.shape
    
    # RGB 색상을 BGR 순서로 변환 (OpenCV는 BGR 형식을 사용함)
    color_bgr = (rgb_color[2], rgb_color[1], rgb_color[0])
    
    # 입력된 RGB 값으로 채워진 단색 이미지 생성
    solid_color_image = np.full((height, width, channels), color_bgr, dtype=np.uint8)
    
    return solid_color_image

def anyline(img_path, out_path, res, blur_b, sigmaColor_b, sigmaSpace_b, blur_a, sigmaColor_a, sigmaSpace_a, line_color, Background, Upscale_tile, use_alpha, show_preview, upscale_lineart, upscale_model, use_path, line_method, use_threshold, threshold_value):
    # 이미지 파일 경로 설정
    image_path = img_path 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 이미지 열기 (알파 채널 포함)
    img = Image.open(image_path).convert("RGBA")
    # 알파 채널 분리
    rgb_img = img.convert("RGB")
    alpha_channel = img.split()[3]
    
    # Background 색상 파싱 (형식: "R,G,B" 또는 "#RRGGBB")
    if Background.startswith('#'):
        # HEX 색상 코드 처리
        bg_color = tuple(int(Background.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    else:
        # R,G,B 형식 처리
        bg_color = tuple(map(int, Background.split(',')))
    
    # 알파 채널을 사용하여 배경색 적용
    img_array = np.array(rgb_img)
    alpha_array = np.array(alpha_channel)
    
    # OpenCV는 BGR 순서를 사용하므로 배경색 순서 변환
    bg_color_bgr = (bg_color[2], bg_color[1], bg_color[0])  # RGB to BGR
    
    # 배경색으로 새 이미지 생성
    background = np.full_like(img_array, bg_color)  # RGB 순서로 배경 생성
    
    # 알파 채널을 0-1 사이 값으로 정규화
    alpha_normalized = alpha_array.astype(float) / 255
    
    # 알파 블렌딩 수행
    for c in range(3):  # RGB 각 채널에 대해
        img_array[:,:,c] = (img_array[:,:,c] * alpha_normalized + 
                           background[:,:,c] * (1 - alpha_normalized))
    
    width, height = img.size
    
    # RGB를 BGR로 변환
    img_array_bgr = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    # Bilateral Filter 적용 a
    filtered_image_before = cv2.bilateralFilter(img_array_bgr, d=blur_b, sigmaColor=sigmaColor_b, sigmaSpace=sigmaSpace_b)  
   
    
    new_width = int(width*math.ceil(res/width))
    new_height = int(height*math.ceil(res/width))
    
    print("Resize image to process : W ", new_width, "/ H ", new_height)
    
    img_resize = cv2.resize(filtered_image_before, (new_width, new_height))
    # 알파 채널도 동일한 크기로 리사이즈
    alpha_resize = alpha_channel.resize((new_width, new_height), Image.Resampling.LANCZOS)
    alpha_resize_np = np.array(alpha_resize)
    
    # BGR을 RGB로 변환하여 PIL 이미지로 변환
    img_resize_rgb = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
    img_filtered = Image.fromarray(img_resize_rgb)
    
    processed_image_anyline = img_filtered
    
    if line_method == 'Anyline':
        # Anyline 모델 불러오기
        anyline = AnylineDetector.from_pretrained("TheMistoAI/MistoLine", filename="MTEED.pth", subfolder="Anyline")
        
        # 객체 감지 수행
        processed_image_anyline = anyline(img_filtered, detect_resolution=res)
    
    if line_method == 'teed':
        # Anyline 모델 불러오기
        teed = TEEDdetector.from_pretrained("fal-ai/teed", filename="5_model.pth")
        
        # 객체 감지 수행
        processed_image_anyline = teed(img_filtered, detect_resolution=res)
        
    if line_method == 'lineart_standard':
        # lineart 모델 불러오기
        lineart = LineartDetector.from_pretrained("lllyasviel/Annotators")
        
        # 객체 감지 수행
        processed_image_anyline = lineart(img_filtered, coarse=True, detect_resolution=res)
        
    if line_method == 'lineart_anime':
        # lineart_anime 모델 불러오기
        lineart_anime = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
        
        # 객체 감지 수행
        processed_image_anyline = lineart_anime(img_filtered, detect_resolution=res)
    
    
    # NumPy 배열로 변환 및 색 반전
    inverted_image = 255 - np.array(processed_image_anyline)
    # Bilateral Filter 적용 b
    filtered_image_after = cv2.bilateralFilter(inverted_image, d=blur_a, sigmaColor=sigmaColor_a, sigmaSpace=sigmaSpace_a)
    
    if use_threshold:
        _, filtered_image_after = cv2.threshold(filtered_image_after, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Line_color 파싱
    line_color_rgb = tuple(map(int, line_color.split(',')))
    
    # 선 색상 변경 (검정색을 Line_color로 변경)
    color_image = create_solid_color_image(filtered_image_after, line_color_rgb)
    
    filtered_image_after = color_blend(filtered_image_after, color_image)

    
    # 결과 시각화
    if show_preview:
        preview_display = cv2.resize(filtered_image_before, (int(512*(width/height)), 512))
        cv2.imshow('Close to proceed', preview_display)
        if use_path:
            cv2.waitKey(100)
        else:
            cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # 파일 이름 설정
    input_image_name = os.path.basename(image_path)
    
    imade_dir = os.path.dirname(image_path)
    
    lineart_folder = os.path.join(out_path, "lineart")
    
    os.makedirs(lineart_folder, exist_ok=True)
    
    output_image_name = os.path.splitext(input_image_name)[0] + ".png"
    
    output_image_path = os.path.join(lineart_folder, output_image_name)
       
    if upscale_lineart :
        temp_image_name = os.path.splitext(input_image_name)[0] + "_temp.png"
        temp_image_path = os.path.join(out_path, temp_image_name)
        cv2.imwrite(temp_image_path, filtered_image_after)
        filtered_image_after = upscale.upscale_image(temp_image_path, out_path, upscale_model, Upscale_tile, False, False)
        os.remove(temp_image_path)
    
    # CV2용 BGRA 이미지 생성
    bgr_image = filtered_image_after
    
    if use_alpha:
        bgra_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2BGRA)
        
        # 알파 채널 크기 확인 후 조정
        if bgra_image.shape[:2] != alpha_resize_np.shape[:2]:
            alpha_resize_np = cv2.resize(alpha_resize_np, (bgra_image.shape[1], bgra_image.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        bgra_image[:, :, 3] = alpha_resize_np  # 알파 채널 설정
        
        save_iamge = bgra_image
        
        preview_image = cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2RGBA)
    else:
        save_iamge = bgr_image
        preview_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    
    # CV2로 저장
    cv2.imwrite(output_image_path, save_iamge)
    print("Save image to : ", output_image_path)

    
    return preview_image