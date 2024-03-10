#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from BaselineRemoval import BaselineRemoval
import tkinter
from sklearn.preprocessing import StandardScaler
import tkinter.messagebox
import customtkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)
import os
import seaborn
from PIL import Image
import random
from tkinter import filedialog
import csv
from scipy.signal import find_peaks
import webbrowser
import joblib


# In[2]:


class StatisticalAnalysisFrame(customtkinter.CTkFrame):
    
    def __init__(self, container):
        
        super().__init__(container)
        
        self.grid_rowconfigure(0 , weight = 5)
        self.grid_rowconfigure(1 , weight = 1)
        self.grid_rowconfigure(2 , weight = 1)
        self.grid_columnconfigure(0 , weight = 1)
        

        # Statistical analysis figure
        
        self.fig = Figure(figsize = (5, 5))
        self.axis = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig,master = self)  
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row = 0 , column = 0 , padx = (5 , 5) , pady = (5 , 5) , sticky="nsew")
        
        # Buttons
        self.buttonFrame = customtkinter.CTkFrame(self)
        self.buttonFrame.grid(row = 1 , column = 0 , padx = 20 , pady = 5)
        
        self.inputButton = customtkinter.CTkButton(self.buttonFrame , text = "Input data" , command = self.inputData)
        self.inputButton.grid(row = 0 , column = 0 , padx = 10)
        
        self.analyseButton = customtkinter.CTkButton(self.buttonFrame , text = "Analyse" , command = self.analyse)
        self.analyseButton.grid(row = 0 , column = 1 , padx = 10)
        
        self.SaveButton = customtkinter.CTkButton(self.buttonFrame , text = "Save figure" , command = self.saveFigure)
        self.SaveButton.grid(row = 0 , column = 2 , padx = 10)
        
        # Label
        self.statLabel = customtkinter.CTkLabel(self , text = "Result Statistic")
        self.statLabel.grid(row = 2 , column = 0 , pady = 5 , padx = 10)
    def inputData(self):
        
        files = tkinter.filedialog.askopenfilenames(parent=self, title='Choose files' , filetypes = [("text files",".txt")])
        
        self.field = list()
        self.intensity = list()
        
        for ele in files:

            values = np.genfromtxt(ele,names=True)
            #print(values['Field'])
            #print(values['index'])
            #print(values['Intensity'])
            self.data=pd.DataFrame(values)
            print(self.data)
    
    
    def analyse(self):
        series=self.data.isnull().sum()
        for index, value in series.items():
            if value!=0:
                col_mean=self.data[index].mean()
                self.data[index]=self.data[index].fillna(col_mean)
        self.data['g-factor']=0.00714471/self.data['Field']  

        self.fig = Figure(figsize=(14, 5))
        
        #self.fig = Figure(figsize = (5, 5))
        #self.axis = self.fig.add_subplot(111)
        
        #self.canvas = FigureCanvasTkAgg(self.fig,master = self)  
        #self.canvas.draw()
        #self.canvas.get_tk_widget().grid(row = 0 , column = 0 , padx = (5 , 5) , pady = (5 , 5) , sticky="nsew")
        #self.axis.clear()
        #self.axis.plot(self.data['Field'] , self.data['g-factor'])
        #self.axis.set_xlabel('Magnetic Field(mT)')
        #self.axis.set_ylabel('g-factor')
        #self.axis.set_title('g-factor v/s Magnetic Field')
        #self.canvas.draw()
        
        self.subplot1 = self.fig.add_subplot(121)  # First subplot
        self.subplot1_visible = True
        self.plot1, = self.subplot1.plot(self.data['Field'], self.data['g-factor'])
        self.subplot1.set_xlabel('Magnetic Field (mT)')
        self.subplot1.set_ylabel('g-factor')
        self.subplot1.set_title('g-factor v/s Magnetic Field')

        self.subplot2 = self.fig.add_subplot(122)  # Second subplot
        self.subplot2_visible = True
        self.plot2, = self.subplot2.plot(self.data['Field'], self.data['Intensity'])
        self.subplot2.set_xlabel('Magnetic Field (mT)')
        self.subplot2.set_ylabel('Intensity')
        self.subplot2.set_title('Intensity v/s Magnetic Field')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row = 0 , column = 0 , sticky="nsew")
        self.input_button = tk.Button(self, text="Input Data", command=self.toggle_plots)
        self.input_button.pack(padx=10, pady=5)
        #print(self.data['g-factor'])
        
    def saveFigure(self):
        
        self.fig.savefig("Plot.png")
          # Open a file dialog for the user to choose the save location and filename
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
          # Save the plot to the user's chosen location and filename
        if file_path:
            self.fig.savefig(file_path)
            tkinter.messagebox.showinfo(title='Save', message='File saved successfully' )
        

    def toggle_plots(self, frame):
        if self.plots_visible:
            frame.plot1.set_visible(False)
            frame.plot2.set_visible(False)
            self.plots_visible = False
        else:
            frame.plot1.set_visible(True)
            frame.plot2.set_visible(True)
            self.plots_visible = True

        frame.canvas.draw()


# In[ ]:


customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class MainFrame(customtkinter.CTkFrame):
    def __init__(self,container):
        
        super().__init__(container)
        
        self.grid_rowconfigure((0 , 1 , 2 , 4) , weight = 0)
        self.grid_rowconfigure(3 , weight = 1)
        self.grid_columnconfigure(0 , weight = 25)
        self.grid_columnconfigure(1 , weight = 0)
        
        # create radiobutton frame
        self.topRightFrame = customtkinter.CTkFrame(self)
        self.topRightFrame.grid(row=0, column=1, rowspan = 2 ,padx=(20, 20), pady=(20, 0), sticky="nsew")
        
        self.inputDataButton = customtkinter.CTkButton(self.topRightFrame , text = "Input Data" , command = self.input_data)
        self.inputDataButton.pack(padx = 20 , pady = 10)
        
        self.preprocessingButton = customtkinter.CTkButton(self.topRightFrame , text = "Data preprocessing" , command = self.preprocessing)
        self.preprocessingButton.pack(padx = 20 , pady = 10)
        
        self.plotButton = customtkinter.CTkButton(self.topRightFrame , text = "Plot" , command = self.plot)
        self.plotButton.pack(padx = 20 , pady = 10)
        
        self.fitGuassianButton = customtkinter.CTkButton(self.topRightFrame , text = " Free Radical Annotation" , command = self.FreeRadical)
        self.fitGuassianButton.pack(padx = 20 , pady = 10)
        
        self.saveCurveButton = customtkinter.CTkButton(self.topRightFrame , text = "Save curve" , command = self.saveCurve)
        self.saveCurveButton.pack(padx = 20 , pady = 10)
        
        # create checkbox and switch frame
        self.bottomRight = customtkinter.CTkFrame(self)
        self.bottomRight.grid(row=2 ,column=1, rowspan = 3 , padx=(20, 20), pady=(20, 20) , sticky = "nsew")
        
        # Image frame
        
        self.imageFrame = customtkinter.CTkFrame(self)
        self.imageFrame.grid(row=0, column=0, rowspan = 4, padx=(20, 0), pady=(20, 20), sticky="nsew")
        
        
        self.fig = Figure(figsize = (5, 5))
        self.axis = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig,master = self.imageFrame)  
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(expand = True , fill = tkinter.BOTH)
        
        # Sliders
        
        self.slider_frame = customtkinter.CTkFrame(self)
        self.slider_frame.grid(row = 4 , column = 0, padx = (20,0) , pady = (0 , 20) , sticky = 'sew')
        
        self.slider_1 = customtkinter.CTkSlider(self.slider_frame, from_= 0, to=1)
        self.slider_1.pack(fill = tkinter.BOTH , expand = True , padx = 20 , pady = 10)
        
        self.slider_2 = customtkinter.CTkSlider(self.slider_frame, from_=0, to=1)
        self.slider_2.pack(fill = tkinter.BOTH, expand = True, padx = 20 , pady = 10)
        
        self.setButton = customtkinter.CTkButton(self.slider_frame , text = 'Set' , command = lambda : self.setXAxis(self.all_smoothed_spectrum , self.slider_1.get() , self.slider_2.get()))
        self.setButton.pack(pady = (0 , 10))
        
        
        # Intitial configuration
        self.plotButton.configure(state = "disabled")
        self.setButton.configure(state="disabled")
        
    def input_data(self):
        
        files = tkinter.filedialog.askopenfilenames(parent=self, title='Choose files' , filetypes = [("text files",".txt"),("csv files",".csv")])
        
        self.field = list()
        self.intensity = list()
        
        for ele in files:

            values = np.genfromtxt(ele,names=True)
            #print(values['Field'])
            #print(values['index'])
            #print(values['Intensity'])
            self.data=pd.DataFrame(values)
            #print(self.data)
    def preprocessing(self):
        series=self.data.isnull().sum()
        for index, value in series.items():
            if value!=0:
                col_mean=self.data[index].mean()
                self.data[index]=self.data[index].fillna(col_mean)
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(self.data)
                tkinter.messagebox.showinfo(title='Save', message='CSV file saved successfully')
        #print(self.data)
        self.setButton.configure(state="enable")
        self.plotButton.configure(state = "enable")
                
    def plot(self):
        #self.canvas = FigureCanvasTkAgg(self.fig,master = self)  
        #self.canvas.draw()
        #self.canvas.get_tk_widget().grid(row = 0 , column = 0 , padx = (5 , 5) , pady = (5 , 5) , sticky="nsew")
        self.axis.clear()
        self.axis.plot(self.data['Field'] , self.data['Intensity'])
        self.axis.set_xlabel('Magnetic Field(mT)')
        self.axis.set_ylabel('Intensity')
        self.axis.set_title('Intensity v/s Magnetic Field')
        self.canvas.draw()
        #self.axis = self.fig.add_subplot(111)
        #self.axis.scatter(self.data['Field'] , self.data['Intensity'])
        #plt.plot(self.data['Field'],self.data['Intensity'])
        #plt.show()
    """
    Find distances and heights of peaks in a spectrum.

    Parameters:
    - spectrum: List of intensity values representing the spectrum.
    - threshold: Threshold parameter for peak detection (default is 0.5).

    Returns:
    - distances: List of distances between adjacent peaks.
    - heights: List of heights of the peaks.
    """
    def FreeRadical(self, threshold=0.5):
        spectrum = self.data['Intensity']
        # Find peaks in the spectrum
        peaks, _ = find_peaks(spectrum, height=threshold)
        # Calculate distances between adjacent peaks
        distances = np.diff(peaks)
        # Get heights of the peaks
        heights = spectrum[peaks]
        
        root = tkinter.Tk()
        root.withdraw()  # Hide the main window
        field = tkinter.simpledialog.askfloat("Magnetic Field Strength", "Enter the magnetic field strength:")
        root.destroy()  # Destroy the hidden window after getting the input
        
        g_factors = heights / (9.274 * (10 ** (-24)) * field)
        splitting_constants = 2.002 * 9.274 * (10 ** (-24)) * distances
        g = ','.join(map(str, g_factors))
        a = ','.join(map(str, splitting_constants))
        
        message = f"G-factors of the peaks: {g}\nHyperfine Splitting Constants of the peaks: {a}"

        # Load the machine learning model
        model = joblib.load("model.pkl")  # Replace "your_model_path.pkl" with the actual path
        
        # Predict radical type using the model
        input_data = np.array([splitting_constants]).T
        #print(input_data)
        radical_type_pred = model.predict([[14.97,14.97,1.00]])
        print(radical_type_pred)
        radical_type=[]
        if radical_type_pred==0 or (len(heights) >= 4 and len(set(heights)) == 1):
            radical_type.append("O2H")
            #message += f"\n\nO2H free radical is present (heights of four consecutive peaks are the same)."

        if radical_type_pred==1 or (len(heights) >= 4 and np.allclose(heights, [heights[0], heights[0]*2, heights[0]*2, heights[0]])):
            radical_type.append("OH")
            #message += f"\n\nOH free radical is present (heights ratio is 1:2:2:1)."
        if radical_type:
            message+=f"\n\nFree radicals {radical_type} are present."
        else:
            message += "\n\nNo specific radicals detected."
                
        tkinter.messagebox.showinfo(title='Free Radical', message=message)
        
   
           
    def saveCurve(self):
        self.fig.savefig("Plot.png")
          # Open a file dialog for the user to choose the save location and filename
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
          # Save the plot to the user's chosen location and filename
        if file_path:
            self.fig.savefig(file_path)
            tkinter.messagebox.showinfo(title='Save', message='File saved successfully' )

        

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("CustomTkinter complex_example.py")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(0, weight = 2)
        self.grid_columnconfigure(1 , weight = 28)
        self.grid_rowconfigure(0 , weight = 1)
        
        # loading image
        image_path = "test_images"
        self.logo_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "Tkinter_logo.jpg")), size=(26, 26))
        self.large_test_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "Tkinter_logo.jpg")), size=(500, 150))
        self.image_icon_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "Tkinter_logo.jpg")), size=(20, 20))
        self.home_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "Tkinter_logo.jpg")),dark_image=Image.open(os.path.join(image_path, "Tkinter_logo.jpg")), size=(20, 20))
        self.chat_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "Tkinter_logo.jpg")),dark_image=Image.open(os.path.join(image_path, "Tkinter_logo.jpg")), size=(20, 20))
        self.add_user_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(image_path, "Tkinter_logo.jpg")),dark_image=Image.open(os.path.join(image_path, "Tkinter_logo.jpg")), size=(20, 20))

        # create sidebar frame with widgets
        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0,column=0 ,sticky="nsw")
        self.navigation_frame.grid_rowconfigure(4 , weight = 1)

        self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame, text="  EPR Analysis", image=self.logo_image,
                                                             compound="left", font=customtkinter.CTkFont(size=15, weight="bold"))
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        self.home_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Home",fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),image=self.home_image, anchor="w", command=self.home_button_event)
        self.home_button.grid(row=1, column=0, sticky="ew")

        self.frame_2_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="g-factor",fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),image=self.chat_image, anchor="w", command=self.frame_2_button_event)
        self.frame_2_button.grid(row=2, column=0, sticky="ew")

        #self.frame_3_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="g-factor",fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),image=self.add_user_image, anchor="w", command=self.frame_3_button_event)
        #self.frame_3_button.grid(row=3, column=0, sticky="ew")

        self.appearance_mode_menu = customtkinter.CTkOptionMenu(self.navigation_frame, values=["Light", "Dark", "System"],command=self.change_appearance_mode_event)
        self.appearance_mode_menu.grid(row=6, rowspan = 3, column=0, padx=20, pady=20, sticky="s")
        

        self.currFrame = MainFrame(self)
        self.currFrame.grid(row = 0 , column = 1,padx = (10,20) , pady=20,sticky = "nsew")
          

    def open_input_dialog_event(self):
        dialog = customtkinter.CTkInputDialog(text="Type in a number:", title="CTkInputDialog")
        print("CTkInputDialog:", dialog.get_input())

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def sidebar_button_event(self):
        print("sidebar_button click")
        
    def home_button_event(self):
            self.currFrame.destroy()
            self.currFrame = MainFrame(self)
            self.currFrame.grid(row=0, column=1, padx=(10, 20), pady=20, sticky="nsew")
    def frame_2_button_event(self):
        self.currFrame.destroy()
        self.currFrame = StatisticalAnalysisFrame(self)  
        self.currFrame.grid(row=0, column=1, padx=(10, 20), pady=20, sticky="nsew")


        
  
    #def frame_3_button_event(self):
        
        #self.currFrame.destroy()
        #self.currFrame = PlotFrame(self)
        #self.currFrame.grid(row = 0 , column = 1,padx = (10,20) , pady=20,sticky = "nsew")
    
    


if __name__ == "__main__":
    app = App()
    app.mainloop()


# In[ ]:




