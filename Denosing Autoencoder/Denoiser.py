import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

# Define the denoising autoencoder function as you provided
def build_denoising_autoencoder(input_shape):
    input_img = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    return autoencoder


# Create the denoising autoencoder
autoencoder = build_denoising_autoencoder(input_shape=(500, 500, 3))

# Initialize Tkinter
root = Tk()
root.title("Denoising Autoencoder App")

# Create a function to open an image file
def open_image():
    global selected_image_reference
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((500, 500))
        img = img.convert('RGB')
        selected_image_reference = ImageTk.PhotoImage(image=img)
        update_image_canvas(selected_image_reference)

# Create a function to denoise the selected image
def denoise_image():
    global selected_image_reference
    if selected_image_reference:
        try:
            img = ImageTk.getimage(selected_image_reference)
            img = img.resize((500, 500))
            img = img.convert('RGB')
            img_array = np.array(img)
            denoised_img = autoencoder.predict(img_array[None, ...])
            denoised_img = denoised_img.squeeze()
            denoised_img = (denoised_img * 255).astype(np.uint8)
            denoised_img = Image.fromarray(denoised_img)
            denoised_image_reference = ImageTk.PhotoImage(image=denoised_img)
            update_image_canvas(denoised_image_reference)
        except Exception as e:
            print(f"Error denoising the image: {str(e)}")

# Create a function to update the image canvas
def update_image_canvas(image_reference):
    canvas.create_image(0, 0, anchor=NW, image=image_reference)
    canvas.image = image_reference

# Create buttons to open and denoise an image
open_button = Button(root, text="Open Image", command=open_image)
denoise_button = Button(root, text="Denoise Image", command=denoise_image)

# Create a canvas to display the image
canvas = Canvas(root, width=500, height=500)


open_button.pack()
denoise_button.pack()
canvas.pack()


selected_image_reference = None
denoised_image_reference = None

# Start the Tkinter main loop
root.mainloop()