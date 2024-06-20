import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import cv2

from keras.models import load_model
model = load_model('model.h5')


classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }

master=tk.Tk()
master.geometry('1280x720')
master.title('Traffic sign classification')
master.configure(background='#CDCDCD')
label=Label(master,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(master)


def classify(image):
    image = image.resize((60,60)).convert("RGB")
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    label.configure(foreground='#011638', text=f"Prediction: {classes[predicted_class]}")

aug = ImageDataGenerator(
    brightness_range=(0.5, 1.5),
    rotation_range=15,
    zoom_range=0.25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest")

def augment_image(image):
    image_data = []
    augmented_image_data = []

    image = cv2.imread(image)
    image_fromarray = Image.fromarray(image, 'RGB')
    resize_image = image_fromarray.resize((60, 60))
    image_data.append(np.array(resize_image))
    image_data = np.array(image_data)

    image = image_data[0]
    augmented_image_data.append(image/255)
    image = np.expand_dims(image, axis=0)

    for _ in range(8):
        augmented_image = aug.flow(image, batch_size=1).next()[0] 
        augmented_image_data.append(augmented_image/255)

    augmented_image_data = np.array(augmented_image_data)

    plt.figure(figsize=(10,6))
    plt.subplot(3, 3, 1)
    plt.imshow(augmented_image_data[0])
    plt.title('Original')
    plt.axis('off')

    for i in range(1, len(augmented_image_data)):
        plt.subplot(3, 3, i+1)
        plt.imshow(augmented_image_data[i])
        plt.title(f'Augmented {i}')
        plt.axis('off')
    plt.show()

def show_classify_button(image):
   classify_b=Button(master,text="Classify Image",command=lambda : classify(image),padx=10,pady=5)
   classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
   classify_b.place(relx=0.79,rely=0.43)

def show_Augumentation_button(image):
   classify_b=Button(master,text="Augument Image",command=lambda : augment_image(image),padx=10,pady=5)
   classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
   classify_b.place(relx=0.79,rely=0.57)

def show_training_history():
    history = np.load('training_history.npy', allow_pickle=True).item()
    pd.DataFrame(history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

def upload_image():
    file_path = filedialog.askopenfilename()
    uploaded = Image.open(file_path)
    im=ImageTk.PhotoImage(uploaded.resize((300, 300)))
    sign_image.configure(image=im)
    sign_image.image=im
    label.configure(text='')

    show_classify_button(uploaded)
    show_Augumentation_button(file_path)


upload=Button(master,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)

show_plot_button = Button(master, text="Show Training History Plot", command=show_training_history, padx=10, pady=5)
show_plot_button.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
show_plot_button.place(relx=0.79,rely=0.50)

sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)

heading = Label(master, text="check traffic sign",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
master.mainloop()