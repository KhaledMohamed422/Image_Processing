import customtkinter as ctk
from image_widgets import *
from PIL import Image, ImageTk
from menu import Menu
from algorithms import adjust_brightness_optimized, cvt2gray_luminance, histogram_equalization, Power_Law_Transformations

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
            'rotate': ctk.DoubleVar(value=ROTATE_DEFAULT),
            'flip': ctk.StringVar(value=FLIP_OPTIONS)
        }
        
        self.tab2_vars = {
            'brightness': ctk.IntVar(value=BRIGHTNESS_DEFAULT),
            'grayscale': ctk.BooleanVar(value=GRAYSCALE_DEFAULT),
            'invert': ctk.BooleanVar(value=INVERT_DEFAULT),
            'hist_eq' : ctk.BooleanVar(value=False),
            'gamma' : ctk.DoubleVar(value=1),
            'histogram_equalization': ctk.BooleanVar(value=False),
            'hist_match_image_path' : ctk.StringVar(value='')
        }
        
        self.tab3_vars = {
            'blur': ctk.DoubleVar(value=BLUR_DEFAULT),
            'contrast': ctk.IntVar(value=CONTRAST_DEFAULT),
            'effect' : ctk.StringVar(value=VIBRANCE_DEFAULT)
        }

        self.tab1_vars['flip'].trace('w',self.process)
        self.tab2_vars['grayscale'].trace('w', lambda*args: self.process('B/W'))
        self.tab2_vars['histogram_equalization'].trace('w', lambda *args: self.process('hist'))

    def process(self, *args):
        self.image = self.original
        print(args)
        processing_func = args[-1]
        
        match processing_func:
            case 'B/W':
                if self.tab2_vars['grayscale'].get() == True:
                    self.image = cvt2gray_luminance(self.image)
                else:
                    self.image = self.original
            case 'Brightness':
                self.image = adjust_brightness_optimized(self.image, self.tab2_vars['brightness'].get())
            case 'hist':
                if self.tab2_vars['histogram_equalization'].get() == True:
                    self.image = histogram_equalization(self.image)
                else:
                    self.image = self.original
            case 'power_trans':
                self.image = Power_Law_Transformations(self.image, self.tab2_vars['gamma'].get())


        
        self.image = self.image.rotate(self.tab1_vars['rotate'].get())

        if self.tab1_vars['flip'].get() == 'X':
            print('X')
        elif self.tab1_vars['flip'].get() == 'Y':
            print('Y')
        elif self.tab1_vars['flip'].get() == 'Both':
            print('Both')
            
        self.place_image()

    def import_image(self, path):
        self.original = Image.open(path)
        self.image = self.original
        self.image_ratio = self.image.size[0] / self.image.size[1]
        self.image_tk = ImageTk.PhotoImage(self.image)
        self.image_import.grid_forget()
        self.image_output = ImageOutput(self, self.resize_image)
        self.close_button = CloseOutput(self, self.close_edit)
        self.menu = Menu(self, self.tab1_vars, self.tab2_vars, self.tab3_vars, self.process, self.export_image)

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
        print(export_string)
        self.image.save(export_string)
App()
