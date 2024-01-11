import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import numpy as np

class ImageProcessorApp:
    def _init_(self, master):
        self.master = master
        self.master.title("Image Processor")
        self.master.configure(bg="#f0f0f0")

        self.image_label = tk.Label(master, bg="#f0f0f0")
        self.image_label.pack(pady=10)

        button_frame = tk.Frame(master, bg="#f0f0f0")
        button_frame.pack(pady=10)

        self.load_button = tk.Button(button_frame, text="Load Image", command=self.load_image, width=15)
        self.load_button.grid(row=0, column=0, padx=10)

        self.rotate_button = tk.Button(button_frame, text="Rotate", command=self.rotate_image, width=15)
        self.rotate_button.grid(row=0, column=1, padx=10)

        self.crop_button = tk.Button(button_frame, text="Crop", command=self.crop_image, width=15)
        self.crop_button.grid(row=0, column=2, padx=10)

        self.flip_button = tk.Button(button_frame, text="Convert to Grayscale", command=self.flip_image, width=20)
        self.flip_button.grid(row=0, column=3, padx=10)

        self.resize_button = tk.Button(button_frame, text="Resize", command=self.resize_image, width=15)
        self.resize_button.grid(row=0, column=4, padx=10)

        self.edge_detect_button = tk.Button(button_frame, text="Edge Detection", command=self.edge_detection, width=15)
        self.edge_detect_button.grid(row=1, column=0, padx=10)



    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            self.display_image()

    def display_image(self):
        if self.image is not None:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(self.image)
            self.tk_image = ImageTk.PhotoImage(image=img_pil)
            self.image_label.configure(image=self.tk_image)
    
    def rotate_image(self):
        if hasattr(self, 'image'):
            rows, cols, _ = self.image.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
            self.image = cv2.warpAffine(self.image, M, (cols, rows))
            self.display_image()

    def crop_image(self):
        if hasattr(self, 'image'):
            self.image = self.image[100:400, 100:400]
            self.display_image()

    def flip_image(self):
        if hasattr(self, 'image'):
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.display_image()

    def resize_image(self):
        if hasattr(self, 'image'):
            user_input = simpledialog.askstring("Resize Image", "Enter width and height (e.g., '300 200'):")
            if user_input:
                try:
                    width, height = map(int, user_input.split())
                    self.image = cv2.resize(self.image, (width, height))
                    self.display_image()
                except ValueError:
                    messagebox.showerror("Error", "Invalid input. Please enter valid width and height.")


    def edge_detection(self):
        if hasattr(self, 'image'):
            self.image = cv2.Canny(self.image, 100, 200)
            self.display_image()


if _name_ == "_main_":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()