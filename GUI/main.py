import customtkinter as ctk
from image_widgets import *
from PIL import Image, ImageTk
from menu import Menu
from algorithms import *

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode('dark')
        ctk.set_default_color_theme("green")
        self.geometry('1000x600')
        self.title('Image Processing')
        self.minsize(800,500)
        self.init_parameters()

        # layout
        self.rowconfigure(0, weight = 1)
        self.columnconfigure(0, weight = 2, uniform = 'a')
        self.columnconfigure(1, weight = 6, uniform = 'a')
        
        self.image_width = 0
        self.image_height = 0
        self.canvas_width = 0
        self.canvas_height = 0

        # widgets
        self.image_import = ImageImport(self, self.import_image) 

        self.mainloop()

    def init_parameters(self):
        self.tab1_vars = {
            'grayscale': ctk.BooleanVar(value=GRAYSCALE_DEFAULT),
            'invert': ctk.BooleanVar(value=INVERT_DEFAULT),
            'histogram_equalization': ctk.BooleanVar(value=False),
            'brightness': ctk.IntVar(value=BRIGHTNESS_DEFAULT),
            'gamma' : ctk.DoubleVar(value=1),
            'image_path' : ctk.StringVar(value=''),
            'drop_options' : ['Histogram Matching', 'Add Image', 'Subtract Image']
        }
        
        self.tab2_vars = {
            'new_min': ctk.IntVar(value=0),
            'new_max': ctk.IntVar(value=255),
            'Draw_Hist':ctk.BooleanVar(value=False),
        }
        
        self.tab3_vars = {
            'reduce': ctk.IntVar(value=1),
            'blur': ctk.IntVar(value=1),
            'Sharp': ctk.BooleanVar(value=False),
            'edge_det': ctk.BooleanVar(value=False),
        }

        self.tab4_vars = {
            'High/Low Pass': ctk.BooleanVar(value=False),
            'Ideal': ctk.DoubleVar(value=0.1),
            'Butterworth': ctk.DoubleVar(value=0.1),
            'Gaussian': ctk.DoubleVar(value=0),
            'order': ctk.IntVar(value=2),
        }

        # tab1
        self.tab1_vars['grayscale'].trace('w', lambda*args: self.process('B/W'))
        self.tab1_vars['invert'].trace('w', lambda*args: self.process('invert'))
        self.tab1_vars['histogram_equalization'].trace('w', lambda *args: self.process('hist'))

        #tab2
        self.tab2_vars['Draw_Hist'].trace('w', lambda*args: self.process('Draw Histogram'))

        #tab3
        self.tab3_vars['edge_det'].trace('w', lambda *args: self.process('Edge'))
        self.tab3_vars['Sharp'].trace('w', lambda *args: self.process('Sharp'))

        #tab4
        self.tab4_vars['High/Low Pass'].trace('w', lambda *args: self.process('High/Low Pass')) 
        
    def process(self, *args):
        self.image = self.original
        print(args)
        processing_func = args[-1]
        
        match processing_func:
            case 'B/W':
                if self.tab1_vars['grayscale'].get() == True:
                    self.image = cvt2gray_luminance(self.image)
                else:
                    self.image = self.original
            case 'invert':
                if self.tab1_vars['invert'].get() == True:
                    self.image = image_negative(self.image)
                else:
                    self.image = self.original
            case 'contrast streching':
                 print(self.tab2_vars['new_min'].get())
                 print(self.tab2_vars['new_max'].get())
                 self.image = Contrast(self.image,self.tab2_vars['new_min'].get(),self.tab2_vars['new_max'].get()) 
               
            case 'Draw Histogram':
                     print(self.image)
                     print(self.tab2_vars['Draw_Hist'].get())
                     if self.tab2_vars['Draw_Hist'].get() == True:
                        Drawing_the_histogram(self.image)
            case 'hist':
                if self.tab1_vars['histogram_equalization'].get() == True:
                    self.image = histogram_equalization(self.image)
                else:
                    self.image = self.original
            case 'Brightness':
                self.image = adjust_brightness_optimized(self.image, self.tab1_vars['brightness'].get())
            case 'power_trans':
                self.image = Power_Law_Transformations(self.image, self.tab1_vars['gamma'].get())
            case 'Histogram Matching':
                self.image = histogram_matching(self.image, self.tab1_vars['image_path'].get())
            case 'Add Image':
                self.image = add_image(self.image, self.tab1_vars['image_path'].get())
            case 'Subtract Image':    
                self.image = sub_image(self.image, self.tab1_vars['image_path'].get())
            case 'blur':
                self.image = Smoothing_Weighted_Filter(self.image, self.tab3_vars['blur'].get())
            case 'Edge':
                    if self.tab3_vars['edge_det'].get() == True:
                        self.image = Edge_Detection(self.image)
                    else:
                         self.image = self.original
            case 'Sharp':
                    if self.tab3_vars['Sharp'].get() == True:
                        self.image = Sharpening_Filter(self.image)
                    else:
                         self.image = self.original
            case 'reduce':
                 self.image = reduce_gray_levels(self.image, self.tab3_vars['reduce'].get()) 
            case 'Ideal':
                if self.tab4_vars['High/Low Pass'].get() == True:
                    print('low') 
                    self.image = ideal_lowpass_filter(self.image, self.tab4_vars['Ideal'].get())
                else:
                    print('High')   
                    self.image = ideal_highpass_filter(self.image, self.tab4_vars['Ideal'].get())
            case 'Butterworth':    
                    if self.tab4_vars['High/Low Pass'].get() == True:
                       print('low') 
                       print(self.tab4_vars['order'].get()) 
                       self.image = Butterworth_lowpass_filter(self.image, self.tab4_vars['Butterworth'].get(),self.tab4_vars['order'].get())
                    else:
                      print('High')   
                      self.image = Butterworth_High_Pass_Filter(self.image, self.tab4_vars['Butterworth'].get(),self.tab4_vars['order'].get())

            case 'Gaussian':    
                    if self.tab4_vars['High/Low Pass'].get() == True:
                       print('Low') 
                       self.image = Butterworth_lowpass_filter(self.image, self.tab4_vars['Gaussian'].get())
                    else:
                      print('High')   
                      self.image = Butterworth_High_Pass_Filter(self.image, self.tab4_vars['Gaussian'].get())                   
            case 'resize':
                    self.image = reverse_1_order(self.image, (self.export_vars['height'].get(), self.export_vars['width'].get()))


        
        #self.image = self.image.rotate(self.tab1_vars['rotate'].get())

        # if self.tab1_vars['flip'].get() == 'X':
        #     print('X')
        # elif self.tab1_vars['flip'].get() == 'Y':
        #     print('Y')
        # elif self.tab1_vars['flip'].get() == 'Both':
        #     print('Both')
            
        self.place_image()

    def import_image(self, path):
        self.original = Image.open(path)
        self.image = self.original
        self.image_ratio = self.image.size[0] / self.image.size[1]
        self.image_tk = ImageTk.PhotoImage(self.image)
        self.image_import.grid_forget()
        self.image_output = ImageOutput(self, self.resize_image)
        self.close_button = CloseOutput(self, self.close_edit)
        self.export_vars = {
            'width' : ctk.IntVar(value=self.original.width),
            'height' : ctk.IntVar(value=self.original.height),
        }
        self.menu = Menu(self, self.tab1_vars, self.tab2_vars, self.tab3_vars, self.tab4_vars, self.export_vars, self.process, self.export_image)

    def close_edit(self):
        # remove the interface
        self.image_output.grid_forget()
        self.close_button.place_forget()
        self.menu.grid_forget()
        self.image_import = ImageImport(self, self.import_image)
        
    def resize_image(self, event):
        canvas_ratio = event.width / event.height

        self.canvas_width = event.width
        self.canvas_height = event.height

        # resize
        if canvas_ratio > self.image_ratio: # canvas is wider than the image
            self.image_height = int(event.height)
            self.image_width = int(self.image_height * self.image_ratio)
        else:
            self.image_width = int(event.width)
            self.image_height = int(self.image_width / self.image_ratio)

        self.place_image()
        # place image
    
    def place_image(self):
        self.image_output.delete('all')
        resized_image = self.image.resize((self.image_width, self.image_height))
        self.image_tk = ImageTk.PhotoImage(resized_image)
        self.image_output.create_image(self.canvas_width / 2, self.canvas_height / 2, image = self.image_tk)
    
    def export_image(self, name, file, path):
        export_string = f'{path}/{name}.{file}'
        if self.export_vars['width'].get() != self.original.width or self.export_vars['height'].get() != self.original.height:
            self.process('resize')
        self.image.save(export_string)
App()
