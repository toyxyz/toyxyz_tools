import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageOps
import numpy as np
import render_3d
import os

class Make3DWindow:
    def __init__(self, parent, image, image_path, path_dir):
        # 새 창 생성
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.title("Make 3D")
        self.window.geometry("800x650")  # 크기 조정
        #self.window.resizable(False, False)
        self.image_path = image_path
        self.path_dir = path_dir
        self.texture_path = None  # 이미지 경로 변수 추가

        # 부모 창 비활성화
        self.parent.attributes("-disabled", True)
        
        # 창 닫힐 때 부모 창 활성화
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # 메인 프레임
        self.main_frame = tk.Frame(self.window)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 왼쪽 이미지 프레임
        self.image_frame = tk.Frame(self.main_frame, width=250, height=250)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        self.image_frame.pack_propagate(False)  # 프레임 크기 고정
        
        # 이미지 레이블
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack(expand=True)
        
        # 프리뷰 이미지 레이블 추가
        self.preview_label = tk.Label(self.image_frame)
        self.preview_label.pack(expand=True)

        # 오른쪽 설정 프레임
        self.config_frame = tk.Frame(self.main_frame)
        self.config_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 파라미터 입력 프레임
        self.param_frame = tk.LabelFrame(self.config_frame, text="Parameters")
        self.param_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 파라미터 기본값
        self.default_params = {
            "input_size": "518",
            "normal_depth": "0.0",
            "normal_min": "0.000",
            "metallic": "0.0",
            "roughness": "1.0",
            "blur": "11",
            "sigmacolor": "75",
            "sigmaspace": "75",
            "Background": "255,255,255",
            "Upscale_tile": "800"
        }
        
        # 파라미터 입력창들
        self.params = {}
        for name, default_value in self.default_params.items():
            frame = tk.Frame(self.param_frame)
            frame.pack(fill=tk.X, padx=5, pady=2)
            
            label = tk.Label(frame, text=f"{name}:", width=10, anchor='w')
            label.pack(side=tk.LEFT, padx=(0, 5))
            
            entry = tk.Entry(frame)
            entry.insert(0, default_value)  # 기본값 설정
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            self.params[name] = entry
            
        # 상단 체크박스들을 담을 새로운 프레임
        top_checks_frame = tk.Frame(self.param_frame)
        top_checks_frame.pack(side=tk.TOP, fill=tk.X)

        # add color 체크박스
        self.enable_color_var = tk.BooleanVar(value=True)
        self.enable_color_check = tk.Checkbutton(
            top_checks_frame, 
            text="Use color texture",
            variable=self.enable_color_var
        )
        self.enable_color_check.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=(5, 0))

        # show bg color 체크박스 추가
        self.show_bg_color_var = tk.BooleanVar(value=False)
        self.show_bg_color_check = tk.Checkbutton(
            top_checks_frame, 
            text="Show BG Color",
            variable=self.show_bg_color_var
        )
        self.show_bg_color_check.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=(5, 0))

        # upsclae normal 체크박스 추가
        self.upscale_normal_var = tk.BooleanVar(value=False)
        self.upscale_normal_check = tk.Checkbutton(
            top_checks_frame, 
            text="Upscale normal",
            variable=self.upscale_normal_var
        )
        self.upscale_normal_check.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=(5, 0))

        # Use path 체크박스 추가
        self.use_path_var = tk.BooleanVar(value=False)
        self.use_path_check = tk.Checkbutton(
            top_checks_frame, 
            text="Use path",
            variable=self.use_path_var
        )
        self.use_path_check.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=(5, 0))

        # Save glTF 체크박스 추가
        self.save_mesh_var = tk.BooleanVar(value=True)
        self.save_mesh_check = tk.Checkbutton(
            self.param_frame, 
            text="Save glTF",
            variable=self.save_mesh_var
        )
        self.save_mesh_check.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=(5, 0))
        
        # 모델 선택 프레임
        self.model_frame = tk.LabelFrame(self.config_frame, text="Depth Model Selection")
        self.model_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 모델 선택 콤보박스
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            self.model_frame, 
            textvariable=self.model_var,
            values=['vits', 'vitb', 'vitl'],
            state='readonly'
        )
        self.model_combo.set('vits')  # 기본값 설정
        self.model_combo.pack(padx=5, pady=5, fill=tk.X)
        
        # 모델 선택 프레임
        self.upscale_model_frame = tk.LabelFrame(self.config_frame, text="Upscale Model Selection")
        self.upscale_model_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 모델 선택 콤보박스
        self.upscale_model_var = tk.StringVar()
        self.upscale_model_combo = ttk.Combobox(
            self.upscale_model_frame, 
            textvariable=self.upscale_model_var,
            values=['RealESRGAN_x4plus', 'RealESRNet_x4plus', 'RealESRGAN_x4plus_anime_6B', 'RealESRGAN_x2plus', 'realesr-animevideov3', 'realesr-general-x4v3'],
            state='readonly'
        )
        self.upscale_model_combo.set('RealESRGAN_x4plus')  # 기본값 설정
        self.upscale_model_combo.pack(padx=5, pady=5, fill=tk.X)
        
        
        # 경로 선택 프레임
        self.path_frame = tk.Frame(self.config_frame)
        self.path_frame.pack(fill=tk.X, pady=(0, 10))

        # 텍스처 프레임 (새로 추가)
        self.texture_frame = tk.Frame(self.path_frame)
        self.texture_frame.pack(fill=tk.X, pady=5)
        
        # 텍스처 경로 프레임 (새로 추가)
        self.texture_dir_frame = tk.Frame(self.path_frame)
        self.texture_dir_frame.pack(fill=tk.X, pady=5)

        # 출력 경로 프레임 (새로 추가)
        self.output_frame = tk.Frame(self.path_frame)
        self.output_frame.pack(fill=tk.X, pady=5)

        # 이미지 추가 버튼
        self.add_texture_button = tk.Button(
            self.texture_frame,  # texture_frame으로 변경
            text="Add Texture",
            command=self.load_image,
            width=10
        )
        self.add_texture_button.pack(side=tk.LEFT, padx=5)

        # 텍스처 경로 표시 라벨
        self.texture_label = tk.Label(self.texture_frame, text="No color Texture selected. Use the default. ", anchor='w')
        self.texture_label.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)
        
        # 텍스쳐 시퀸스 경로 선택 버튼
        self.texture_dir = None
        self.texture_dir_button = tk.Button(
            self.texture_dir_frame,  # output_frame으로 변경
            text="Select Texture directory",
            command=self.select_texture_directory
        )
        self.texture_dir_button.pack(side=tk.LEFT, padx=5)

        # 텍스쳐 시퀸스 경로 표시 라벨
        self.texture_dir_label = tk.Label(self.texture_dir_frame, text="No Texture Directory selected", anchor='w')
        self.texture_dir_label.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)
        
        
        # 경로 선택 버튼
        self.out_dir = None
        self.path_button = tk.Button(
            self.output_frame,  # output_frame으로 변경
            text="Select Output Directory",
            command=self.select_output_directory
        )
        self.path_button.pack(side=tk.LEFT, padx=5)

        # 경로 표시 라벨
        self.path_label = tk.Label(self.output_frame, text="No directory selected. Use the default path /output. ", anchor='w')
        self.path_label.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)
        
        # 실행 버튼
        self.run_button = tk.Button(
            self.config_frame,
            text="Generate 3D",
            command=self.generate_3d
        )
        self.run_button.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=5)
        
        # 이미지 표시
        if image:
            self.display_image(image)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            image = Image.open(file_path)
            self.current_image = image
            self.texture_path = file_path  # 이미지 경로 저장
            self.texture_label.config(text=self.texture_path)  # 경로 라벨에 선택된 경로 표시        
            print("Use texture:", self.texture_path)
            
        # 경로 선택 후 최상단 설정 다시 적용
        self.window.lift()
        self.window.focus_force()

    def display_image(self, image):
        # 이미지 크기 조정
        display_size = (230, 230)  # 여백을 위해 프레임보다 작게 설정
        image_copy = image.copy()
        image_copy.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        # PhotoImage로 변환
        photo = ImageTk.PhotoImage(image_copy)
        
        # 레이블에 이미지 표시
        self.image_label.configure(image=photo)
        self.image_label.image = photo
        
    def display_preview_image(self, image):
        # NumPy 배열을 PIL 이미지로 변환
        if isinstance(image, np.ndarray):
            # 배열 데이터 타입 변환
            image = image[..., 0]
            image = image.astype(np.uint8)  # 값 범위를 [0, 255]로 변환
            image = Image.fromarray(image)

        # 미리보기 이미지 크기 조정
        display_size = (230, 230)  # 여백을 위해 프레임보다 작게 설정
        image_copy = image.copy()
        image_copy.thumbnail(display_size, Image.Resampling.LANCZOS)

        # PhotoImage로 변환
        preview_photo = ImageTk.PhotoImage(image_copy)

        # 레이블에 미리보기 이미지 표시 (원본 이미지 대신 교체)
        self.image_label.configure(image=preview_photo)
        self.image_label.image = preview_photo
        
    def select_output_directory(self):
        # 경로 선택 대화 상자 열기
        directory = filedialog.askdirectory()
        if directory:
            self.out_dir = directory  # 선택된 경로 저장
            self.path_label.config(text=self.out_dir)  # 경로 라벨에 선택된 경로 표시
            print("Output directory selected:", self.out_dir)
        
        # 경로 선택 후 최상단 설정 다시 적용
        self.window.lift()
        self.window.focus_force()
        
    def select_texture_directory(self):
        # 텍스쳐 경로 선택 대화 상자 열기
        directory = filedialog.askdirectory()
        if directory:
            self.texture_dir = directory  # 선택된 경로 저장
            self.texture_dir_label.config(text=self.texture_dir)  # 경로 라벨에 선택된 경로 표시
            print("Texture directory selected:", self.texture_dir)
        
        # 경로 선택 후 최상단 설정 다시 적용
        self.window.lift()
        self.window.focus_force()
        
    def generate_3d(self):
        # 파라미터 값들 가져오기
        params = {name: entry.get() for name, entry in self.params.items()}
        selected_model = self.model_var.get()
        
        if self.out_dir == None :
            self.out_dir = "output"
        
        # 3D 생성
        print("Generating 3D with parameters:", params)
        print("Add color texture:", self.enable_color_var.get())
        print("Selected model:", selected_model)
        print("Output directory:", self.out_dir)
        if self.use_path_var.get():
            if self.path_dir != None:
                print("Image path :", self.path_dir)
        else :
            if self.image_path != None:
                print("Image path:", self.image_path)
        print("Show Background color:", self.show_bg_color_var.get())
        print("Upscale Normal:", self.upscale_normal_var.get())
        if self.upscale_normal_var.get():
            print("Upscale Model:", self.upscale_model_var.get())
        print("Use Path:", self.use_path_var.get())
        print("Save gITF:", self.save_mesh_var.get())
        if self.texture_path is not None :
            print("Use Color texture:", self.texture_path)
        
        print("===========================================Gernerate 3D ===================================================")
        
        if self.use_path_var.get():
            # path_dir 내 모든 이미지 파일 순회
            if self.path_dir is not None :
                
                # 이미지 파일만 필터링
                image_files = [filename for filename in sorted(os.listdir(self.path_dir)) 
                               if os.path.isfile(os.path.join(self.path_dir, filename)) 
                               and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                
                # 이미지 파일이 없는 경우 메시지 출력
                if not image_files:
                    print("No images in path!")
                
                else:
                    for filename in sorted(os.listdir(self.path_dir)):
                        file_path = os.path.join(self.path_dir, filename)
                        if self.texture_dir is not None:
                            texture_path = os.path.join(self.texture_dir, filename)
                        else :
                            texture_path = None
                              
                        if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            print("Processing image:", file_path)

                            preview_3d = render_3d.render_depth_normal_mesh(
                                file_path, 
                                int(self.params["input_size"].get()), 
                                self.out_dir, 
                                float(self.params["normal_depth"].get()), 
                                float(self.params["normal_min"].get()), 
                                float(self.params["metallic"].get()), 
                                float(self.params["roughness"].get()), 
                                int(self.params["blur"].get()), 
                                float(self.params["sigmacolor"].get()), 
                                float(self.params["sigmaspace"].get()), 
                                selected_model,
                                str(self.params["Background"].get()),
                                self.enable_color_var.get(),
                                self.show_bg_color_var.get(),
                                self.upscale_normal_var.get(),
                                str(self.upscale_model_var.get()),
                                self.save_mesh_var.get(),
                                self.use_path_var.get(),
                                int(self.params["Upscale_tile"].get()),
                                texture_path
                            )

                            # 미리보기 이미지 표시
                            self.display_preview_image(preview_3d)  # 생성된 미리보기 이미지 표시
            else :
                print("No image path! Please Add image path")
        else :
            if self.image_path :
                preview_3d = render_3d.render_depth_normal_mesh(
                    str(self.image_path), 
                    int(self.params["input_size"].get()), 
                    self.out_dir, 
                    float(self.params["normal_depth"].get()), 
                    float(self.params["normal_min"].get()), 
                    float(self.params["metallic"].get()), 
                    float(self.params["roughness"].get()), 
                    int(self.params["blur"].get()), 
                    float(self.params["sigmacolor"].get()), 
                    float(self.params["sigmaspace"].get()), 
                    selected_model,
                    str(self.params["Background"].get()),
                    self.enable_color_var.get(),
                    self.show_bg_color_var.get(),
                    self.upscale_normal_var.get(),
                    str(self.upscale_model_var.get()),
                    self.save_mesh_var.get(),
                    self.use_path_var.get(),
                    int(self.params["Upscale_tile"].get()),
                    self.texture_path
                )
                
                # 미리보기 이미지 표시
                self.display_preview_image(preview_3d)  # 생성된 미리보기 이미지 표시
            else :
                print("No image! Please Add image")

    def on_close(self):
        # 창을 닫을 때 부모 창 활성화
        self.parent.attributes("-disabled", False)
        self.window.destroy()

class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Toy Tools")
        
        self.image_path = None  # 이미지 경로 변수 추가
        
        self.open_dir = None # 이미지 폴더 경로 변수 추가
        
        # 윈도우 크기 설정
        self.root.geometry("800x600")
        
        # 메인 프레임 생성
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 우측 버튼 프레임
        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        # 이미지 추가 버튼
        self.add_button = tk.Button(
            self.button_frame,
            text="Add Image",
            command=self.load_image,
            width=10
        )
        self.add_button.pack(side=tk.TOP, pady=(0, 5))
        
        # 이미지 경로 추가 버튼
        self.add_path_button = tk.Button(
            self.button_frame ,
            text="Add Path",
            command=self.select_open_directory,
            width=10
        )
        self.add_path_button.pack(side=tk.TOP, pady=(0, 5))
        
        # Make 3D 버튼
        self.make_3d_button = tk.Button(
            self.button_frame,
            text="Make 3D",
            command=self.make_3d,
            width=10,
            state=tk.DISABLED
        )
        self.make_3d_button.pack(side=tk.TOP)
        
        # 이미지 프레임
        self.image_frame = tk.Frame(self.main_frame)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.open_path_label = tk.Label(self.image_frame, text="No image path selected. ", anchor='w')
        self.open_path_label.pack(side=tk.TOP, padx=(10, 0), fill=tk.X)
        
        # 이미지 레이블
        self.image_label = tk.Label(self.image_frame, text="No image")
        self.image_label.pack(expand=True)
        
        self.current_image = None  # 현재 이미지 변수 추가
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            image = Image.open(file_path)
            self.current_image = image
            self.image_path = file_path  # 이미지 경로 저장
            
            self.root.after(100, self.update_image)
            self.make_3d_button.configure(state=tk.NORMAL)
            
            print("Open image:", self.image_path)
            
    def update_image(self):
        if self.current_image:
            resized_image = self.resize_image(self.current_image)
            photo = ImageTk.PhotoImage(resized_image)
            self.image_label.configure(image=photo, text="")  # 이미지 표시 시 텍스트 제거
            self.image_label.image = photo
        else:
            self.image_label.configure(image="", text="No image")  # 이미지 없을 때 텍스트 표시
            
    def resize_image(self, image):
        frame_width = self.image_frame.winfo_width()
        frame_height = self.image_frame.winfo_height()
        
        img_width, img_height = image.size
        
        width_ratio = frame_width / img_width
        height_ratio = frame_height / img_height
        
        ratio = min(width_ratio, height_ratio)
        new_width = int(img_width * ratio * 0.9)
        new_height = int(img_height * ratio * 0.9)
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
    def make_3d(self):
        if self.image_path or self.open_dir:
            Make3DWindow(self.root, self.current_image, self.image_path, self.open_dir)
            
    def select_open_directory(self):
        # 경로 선택 대화 상자 열기
        directory = filedialog.askdirectory()
        if directory:
            self.open_dir = directory  # 선택된 경로 저장
            self.open_path_label.config(text=self.open_dir)  # 경로 라벨에 선택된 경로 표시
            self.make_3d_button.configure(state=tk.NORMAL)
            print("Image Path selected:", self.open_dir)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageViewer(root)
    root.mainloop()
