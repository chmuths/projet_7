
# coding: utf-8

# ### Image labelling using CNN

# This notebook does a transfer learning based on the VGG-16 pre-trained network.

# In[1]:


# Import libraries and modules
import sys

import numpy as np
import pandas as pd
import pickle

import PIL.Image
from PIL import ImageTk

from keras.preprocessing.image import load_img as k_load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

from tkinter import *
import tkinter.filedialog


# ### Utility methods

# #### Reading pictures and storing them into an array

# In[2]:


# Method to read a file picture and return it as VGG-16 requires
def load_img_file(picture_file):
    # Load picture and resez all to 224x224 pixels
    img = k_load_img(picture_file, target_size=(224, 224))
    
    # Picture has to be converted into Numpy array
    img = img_to_array(img)
    
    return img


# In[3]:


def predict_breed_img(img_file):
    img = load_img_file(img_file)

    # Preprocess input as VGG-16 requires it
    new_picture = preprocess_input(img)
    new_picture = new_picture.reshape(1, 3, 224, 224)

    # Normalize all values to be between 0 and 1
    new_picture = new_picture.astype('float32')

    new_picture /= 255

    # Evaluate model on test data
    new_label = model.predict(new_picture)

    predicted_breed = np.argmax(new_label)
    
    return predicted_breed


# In[4]:


class UI(Frame):
    
    def __init__(self, fenetre, **kwargs):
        Frame.__init__(self, fenetre, width=768, height=576, **kwargs)
        self.pack(fill=BOTH)
        
        self.titre = Label(self, text="Reconnaissance de race de chiens", font=("Arial", 18))
        self.titre.pack()
        
        self.bouton = Button(self, text="Sélectionnez une image", font=("Arial", 12), command=self.predict_breed )
        self.bouton.pack()
        
        # Create area to display the loaded image
        self.canvas = Canvas(self, width=350, height=350)
        self.image_dog = self.canvas.create_image(0, 0, anchor=NW)
        self.canvas.pack()
        
        self.breed = Label(self, text="Réponse ici", font=("Arial", 14))
        self.breed.pack()
        
        self.nb = Label(self, text="Base {0} races".format(nb_breeds), font=("Arial", 8))
        self.nb.pack()

    
    def predict_breed(self):
        self.breed["text"] = "Veuillez patienter..."
        # Load image
        picture_file = tkinter.filedialog.askopenfilename()

        predicted_breed = predict_breed_img(picture_file)
        
        # Load image to display it
        pil_img = PIL.Image.open(picture_file)
        
        # Resize image to fit into the canvas size
        wpercent = (350/float(pil_img.size[0]))
        hsize = int((float(pil_img.size[1])*float(wpercent)))
        pil_img = pil_img.resize((350,hsize), PIL.Image.ANTIALIAS)
        
        # Convert image into tkinter compatible format and display it
        myImage = ImageTk.PhotoImage(pil_img)
        self.myImage = myImage  # Avoid Python garbage collector to remove image data
        self.canvas.itemconfig(self.image_dog, image = myImage)
               
        self.breed["text"] = ("Ce chien est probablement de race {0} "
                              .format(df_breeds.loc[predicted_breed, ['breed_name']].values[0]))


# ### Main execution

# In[5]:


if __name__ == "__main__":

    # Number of dogs breeds to include
    nb_breeds = 10

    print("Nombre d'arguments {0}".format(len(sys.argv)))
    for arg in sys.argv:
        print(arg)

    if len(sys.argv) == 2:
        nbb = sys.argv[1]
        if nbb in ("2", "10", "30", "50"):
            nb_breeds = int(nbb)

    df_breeds = pd.read_csv("breeds_list.csv")

    # restore the model
    model_name = 'vgg14-{0}-{1}-{2}.pkl'.format(nb_breeds, 'full', '20')
    # print('Restoring model {0}'.format(model_name))
    input_file = open(model_name, 'rb')
    model = pickle.load(input_file)
    input_file.close()

    # Start graphical User Interface

    my_window = Tk()
    interface = UI(my_window)
    interface.mainloop()

