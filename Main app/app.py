import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import Scale
from tkinter import filedialog
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model


class ImageProcessorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Processor")
        self.master.configure(bg="#f0f0f0")

        self.tabs = ttk.Notebook(master)
        self.tabs.pack(fill='both', expand=True)

        
        self.tab1 = tk.Frame(self.tabs)
        self.tab2 = tk.Frame(self.tabs)
        self.tab3 = tk.Frame(self.tabs)
        self.tab4 = tk.Frame(self.tabs)
        

        self.tabs.add(self.tab1, text="Image processing")
        self.tabs.add(self.tab2, text="Advanced")
        self.tabs.add(self.tab3, text="Style Transfer")
        self.tabs.add(self.tab4, text="Image Segmentation")


        self.image_label = tk.Label(master, bg="#f0f0f0")
        self.image_label.pack(pady=10)

        button_frame = tk.Frame(master, bg="#f0f0f0")
        button_frame.pack(pady=10)

        self.load_button = tk.Button(self.tab1, text="Load Image", command=self.load_image, width=15)
        self.load_button.grid(row=0, column=0, padx=10, pady=10)

        self.rotate_button = tk.Button(self.tab1, text="Rotate", command=self.rotate_image, width=15)
        self.rotate_button.grid(row=0, column=1, padx=10, pady=10)

        self.crop_button = tk.Button(self.tab1, text="Crop", command=self.crop_image, width=15)
        self.crop_button.grid(row=0, column=2, padx=10, pady=10)

        self.flip_button = tk.Button(self.tab1, text="Convert to Grayscale", command=self.flip_image, width=20)
        self.flip_button.grid(row=0, column=3, padx=10, pady=10)

        self.resize_button = tk.Button(self.tab1, text="Resize", command=self.resize_image, width=15)
        self.resize_button.grid(row=0, column=4, padx=10, pady=10)

        self.reset_button = tk.Button(self.tab1, text="Reset Image", command=self.reset_image, width=15)
        self.reset_button.grid(row=0, column=7, padx=10, pady=10)

        self.sharpen_label = tk.Label(self.tab2, text="Sharpen Image", width=20)
        self.sharpen_label.grid(row=0, column=0, padx=10, pady=10)
        self.sharpen_slider = Scale(self.tab2, from_=0, to=2, resolution=0.1, orient="horizontal", length=200)
        self.sharpen_slider.set(1.0)  # Set the initial value
        self.sharpen_slider.grid(row=0, column=1, padx=10, pady=10)

        self.sharpen_button = tk.Button(self.tab2, text="Apply Sharpen", command=self.sharpen_image, width=20)
        self.sharpen_button.grid(row=0, column=2, pady=10)




        self.tonal_label = tk.Label(self.tab2, text="Tonal Transformation", width=20)
        self.tonal_label.grid(row=1, column=0, padx=10, pady=10)
        self.tonal_slider = Scale(self.tab2, from_=1, to=3, resolution=0.1, orient="horizontal", length=200)
        self.tonal_slider.set(2.0)  # Set the initial value
        self.tonal_slider.grid(row=1, column=1, pady=10)

        self.tonal_transform_button = tk.Button(self.tab2, text="Apply Tonal Transformation", command=self.tonal_transform, width=20)
        self.tonal_transform_button.grid(row=1, column=2, padx=10, pady=10)

        self.smooth_button = tk.Button(self.tab2, text="Blur", command=self.blur_image, width=15)
        self.smooth_button.grid(row=2, column=1, padx=10)

        self.edge_detect_button = tk.Button(self.tab2, text="Edge Detection", command=self.edge_detection, width=15)
        self.edge_detect_button.grid(row=2, column=2, padx=10)
        
        self.point_detect_button = tk.Button(self.tab2, text="Point Detect", command=self.sift_feature_detection, width=15)
        self.point_detect_button.grid(row=2, column=3, padx=10)

        self.load_style_button = tk.Button(self.tab3, text="Load Style Image", command=self.load_style_image)
        self.load_style_button.grid(row=2, column=5, pady=5)

        self.stylize_button = tk.Button(self.tab3, text="Stylize", command=self.stylize_images)
        self.stylize_button.grid(row=2, column=6, pady=5)

        self.load_content_button = tk.Button(self.tab3, text="Load Content Image", command=self.load_content_image)
        self.load_content_button.grid(row=2, column=4, pady=5)


        self.load_button = tk.Button(self.tab4, text="Load Image", command=self.load_image, width=15)
        self.load_button.grid(row=0, column=0, padx=10, pady=10)

        self.segmentation_button = tk.Button(self.tab4, text="Segment Image", command=self.segment_image, width=15)
        self.segmentation_button.grid(row=0, column=1)

        self.model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
        self.segmentation_model = self.build_segmentation_model()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            self.original_image = self.image.copy()
            self.display_image()

    def reset_image(self):
        if self.original_image is not None:
            self.image = self.original_image.copy()
            self.display_image()

    def load_style_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.style_image = self.load_image2(file_path)
            self.display_image2(self.style_image)

    def load_content_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.content_image = self.load_image2(file_path)
            self.display_image2(self.content_image)

    def load_image2(self, img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = img[tf.newaxis, :]
        return img
    
    def display_image2(self, image):
        if image is not None:
            img = np.squeeze(image)
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)
            img_pil = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(img_pil)
            self.image_label.configure(image=img_tk)
            self.image_label.image = img_tk

    def display_image(self):
        if self.image is not None:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(self.image)
            self.tk_image = ImageTk.PhotoImage(image=img_pil)
            self.image_label.configure(image=self.tk_image)

    def stylize_images(self):
        if self.content_image is not None and self.style_image is not None:
            self.stylized_image = self.model(tf.constant(self.content_image), tf.constant(self.style_image))[0]
            self.display_image2(self.stylized_image)
            
    
    def rotate_image(self):
        if hasattr(self, 'image'):
            rows, cols, _ = self.image.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
            self.image = cv2.warpAffine(self.image, M, (cols, rows))
            self.display_image()

    def crop_image(self):
        if hasattr(self, 'image'):
            if self.image is not None:
                original_height, original_width, _ = self.image.shape
                crop_coordinates = simpledialog.askstring("Crop Image", f"Enter crop coordinates and dimensions (x y width height)\nOriginal dimensions: {original_width}x{original_height}:")
                
                if crop_coordinates:
                    try:
                        x, y, width, height = map(int, crop_coordinates.split())
                        if 0 <= x < original_width and 0 <= y < original_height and x + width <= original_width and y + height <= original_height:
                            self.image = self.image[y:y+height, x:x+width]
                            self.display_image()
                        else:
                            messagebox.showerror("Error", "Invalid crop coordinates. Please ensure they are within the image bounds.")
                    except ValueError:
                        messagebox.showerror("Error", "Invalid input. Please enter valid coordinates and dimensions.")

    def flip_image(self):
        if hasattr(self, 'image'):
            og_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

           
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(og_image, cmap='gray')
            axs[0].set_title('Original Image')
            axs[0].axis('off')
            
            axs[1].imshow(gray_image, cmap='gray')
            axs[1].set_title('Grayscale Image')
            axs[1].axis('off')
            
            plt.show()
    
    def resize_image(self):
        if hasattr(self, 'image'):
            og_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            original_height, original_width, _ = og_image.shape  # Get original dimensions
            user_input = simpledialog.askstring("Resize Image", f"Enter width and height (e.g., '300 200')\nOriginal dimensions: {original_width}x{original_height}:")
            if user_input:
                try:
                    width, height = map(int, user_input.split())
                    resize_image = cv2.resize(self.image, (width, height))
                    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                    axs[0].imshow(og_image, cmap='gray')
                    axs[0].set_title('Original Image')
                    
            
                    axs[1].imshow(resize_image, cmap='gray')
                    axs[1].set_title('Resized Image')
                

                    plt.show()
                except ValueError:
                    messagebox.showerror("Error", "Invalid input. Please enter valid width and height.")

    def sharpen_image(self):
        if hasattr(self, 'image'):
            og_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            sharpen_factor = float(self.sharpen_slider.get())
            kernel = np.array([[-1, -1, -1],
                               [-1, 9 + sharpen_factor, -1],
                               [-1, -1, -1]])
            sharpened_image = cv2.filter2D(self.image, -1, kernel)
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(og_image, cmap='gray')
            axs[0].set_title('Original Image')
            axs[0].axis('off')
            
            axs[1].imshow(sharpened_image, cmap='gray')
            axs[1].set_title(f'Sharpened Image (Factor: {sharpen_factor})')
            axs[1].axis('off')
            
            plt.show()

    def blur_image(self):
        if hasattr(self, 'image'):
            og_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            blur_image = cv2.GaussianBlur(self.image, (15, 15), 0)
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(og_image, cmap='gray')
            axs[0].set_title('Original Image')
            axs[0].axis('off')
            
            axs[1].imshow(blur_image, cmap='gray')
            axs[1].set_title('Blurred Image')
            axs[1].axis('off')
            
            plt.show()

    def edge_detection(self):
        if hasattr(self, 'image'):
            og_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            edge_image = cv2.Canny(self.image, 100, 200)
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(og_image, cmap='gray')
            axs[0].set_title('Original Image')
            axs[0].axis('off')
            
            axs[1].imshow(edge_image, cmap='gray')
            axs[1].set_title('Detected Edges')
            axs[1].axis('off')
            
            plt.show()

    

    def tonal_transform(self):
        if hasattr(self, 'image'):
            og_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            tonal_factor = float(self.tonal_slider.get())
            tonal_image = cv2.convertScaleAbs(self.image, alpha=tonal_factor, beta=0)
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(og_image, cmap='gray')
            axs[0].set_title('Original Image')
            axs[0].axis('off')
            
            axs[1].imshow(tonal_image, cmap='gray')
            axs[1].set_title(f'Tonal Transformed (x{tonal_factor}) Image')
            axs[1].axis('off')
            
            plt.show()

    def sift_feature_detection(self):
        if hasattr(self, 'image'):
            og_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT_create()
            keypoints = sift.detect(gray_image, None)
            sift_image = cv2.drawKeypoints(self.image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(og_image, cmap='gray')
            axs[0].set_title('Original Image')
            axs[0].axis('off')
            
            axs[1].imshow(sift_image, cmap='gray')
            axs[1].set_title('Point detected Image')
            axs[1].axis('off')
            
            plt.show()


    def build_segmentation_model(self):
        input_size = (128, 128, 3)

        inputs = Input(input_size)
        
        # U-Net architecture
        # Encoder
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
        # Bottleneck
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    
        # Decoder
        up4 = UpSampling2D(size=(2, 2))(conv3)
        concat4 = Concatenate()([conv2, up4])
        conv4 = Conv2D(128, 3, activation='relu', padding='same')(concat4)
        up5 = UpSampling2D(size=(2, 2))(conv4)
        concat5 = Concatenate()([conv1, up5])
        conv5 = Conv2D(64, 3, activation='relu', padding='same')(concat5)
    
        outputs = Conv2D(1, 1, activation='sigmoid')(conv5)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def segment_image(self):
        if hasattr(self, 'image'):
            input_image = cv2.resize(self.image, (128, 128))
            input_image = input_image / 255.0  # Normalize

           
            segmentation_mask = self.segmentation_model.predict(np.expand_dims(input_image, axis=0))[0]

            
            threshold = 0.5  # You can adjust this threshold
            binary_mask = (segmentation_mask > threshold).astype(np.uint8) * 255

           
            self.image = binary_mask
            self.display_image()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
