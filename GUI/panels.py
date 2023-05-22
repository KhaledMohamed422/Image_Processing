import customtkinter as ctk
from tkinter import filedialog
from settings import *
from os.path import basename

class Panel(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(master = parent, fg_color = DARK_GREY)
        self.pack(fill = 'x', pady = 4, ipady = 8)

class SliderPanel(Panel):
    def __init__(self, parent, text, data_var,  min_value, max_value, func):
        super().__init__(parent = parent)

        # layout 
        self.rowconfigure((0,1,2), weight = 1)
        self.columnconfigure((0,1), weight = 1)

        self.data_var = data_var
        self.data_var.trace('w', self.update_text)

        ctk.CTkLabel(self, text = text).grid(column = 0, row = 0, sticky = 'W', padx = 5)
        self.num_label = ctk.CTkLabel(self, text = data_var.get())
        self.num_label.grid(column = 1, row = 0, sticky = 'E', padx = 5)
        ctk.CTkSlider(self,
                fg_color = SLIDER_BG,
                variable = self.data_var, 
                from_ = min_value,
                to = max_value).grid(row = 1, column = 0, columnspan = 2, sticky = 'ew', padx = 5, pady = 5)
        self.apply_button = ctk.CTkButton(self, text = "Apply", command = func)
        self.apply_button.grid(column = 0, columnspan = 2, row = 2, sticky='ew')

    def update_text(self,*args):
        self.num_label.configure(text = f'{round(self.data_var.get(), 2)}')

class SliderPanelWithSteps(Panel):
    def __init__(self, parent, text, data_var,  min_value, max_value, func,step):
        super().__init__(parent = parent)

        # layout 
        self.rowconfigure((0,1,2), weight = 1)
        self.columnconfigure((0,1), weight = 1)

        self.data_var = data_var
        self.data_var.trace('w', self.update_text)

        ctk.CTkLabel(self, text = text).grid(column = 0, row = 0, sticky = 'W', padx = 5)
        self.num_label = ctk.CTkLabel(self, text = data_var.get())
        self.num_label.grid(column = 1, row = 0, sticky = 'E', padx = 5)
        ctk.CTkSlider(self,
                fg_color = SLIDER_BG,
                variable = self.data_var, 
                from_ = min_value,
                to = max_value,
                number_of_steps=step).grid(row = 1, column = 0, columnspan = 2, sticky = 'ew', padx = 5, pady = 5)
        self.apply_button = ctk.CTkButton(self, text = "Apply", command = func)
        self.apply_button.grid(column = 0, columnspan = 2, row = 2, sticky='ew')

    def update_text(self,*args):
        self.num_label.configure(text = f'{round(self.data_var.get(), 2)}')
        
class OneEntryPanel(Panel):
    def __init__(self, parent, maintext, text1, var1):
        super().__init__(parent = parent)
        
        self.rowconfigure((0,1,2), weight = 1)
        self.columnconfigure((0,1), weight = 1)
        
        ctk.CTkLabel(self, text = maintext).grid(column = 0, row = 0, columnspan = 2, padx = 5, sticky = "W")
        
        ctk.CTkLabel(self, text = text1).grid(column = 0, row = 1, sticky = 'W', padx = 5)
        ctk.CTkEntry(self, textvariable = var1).grid(column = 1, row = 1, sticky='E')
 
class TwoEntryPanel(Panel): 
    def __init__(self, parent, maintext, text1, text2, var1, var2, func):
        super().__init__(parent = parent)
        
        self.rowconfigure((0,1,2,3), weight = 1)
        self.columnconfigure((0,1), weight = 1)
        
        ctk.CTkLabel(self, text = maintext).grid(column = 0, row = 0, columnspan = 2, sticky = 'W', padx = 5)
        
        ctk.CTkLabel(self, text = text1).grid(column = 0, row = 1, sticky = 'W', padx = 5)
        ctk.CTkLabel(self, text = text2).grid(column = 1, row = 1, sticky = 'W', padx = 5)

        ctk.CTkEntry(self, textvariable = var1).grid(column = 0, row = 2)
        ctk.CTkEntry(self, textvariable = var2).grid(column = 1, row = 2)
        self.apply_button = ctk.CTkButton(self, text = "Apply", command = func)
        self.apply_button.grid(column = 0, columnspan = 2, row = 3, sticky='ew')
      
class CustomTwoEntryPanel(TwoEntryPanel):
    def __init__(self, parent, maintext, text1, text2, var1, var2, func):
        super().__init__(parent=parent, maintext=maintext, text1=text1, text2=text2, var1=var1, var2=var2, func=func)

        # Custom modifications
        self.apply_button.destroy()  # Remove the Apply button

class SegmentPanel(Panel):
    def __init__(self, parent, text, data_vars, options):
        super().__init__(parent = parent)

        ctk.CTkLabel(self, text = text).pack()
        ctk.CTkSegmentedButton(self,variable=data_vars, values = options).pack(expand = True, fill = 'both', padx = 4, pady = 4)

class SwitchPanel(Panel):
    def __init__(self, parent, *args): # (var, text)
        super().__init__(parent=parent)
        for var, text in args:
            switch = ctk.CTkSwitch(self, text = text, variable=  var, button_color = GREEN, fg_color = SLIDER_BG)
            switch.pack(side = 'left', expand = True, fill = 'both', padx = 5, pady = 5)

class DropDownPanel(ctk.CTkOptionMenu):
    def __init__(self, parent, data_vars, options): # (var, text)
        super().__init__(
                master=parent,
                values = options,
                fg_color = DARK_GREY,
                button_color = DROPDOWN_MAIN_COLOR,
                button_hover_color = DROPDOWN_HOVER_COLOR,
                dropdown_fg_color = DROPDOWN_MENU_COLOR,
                variable = data_vars)

        self.pack(fill = 'x', pady = 4)

class RevertButton(ctk.CTkButton):
    def __init__(self, parent, *args):
        super().__init__(master = parent, text = 'Revert', command = self.revert)
        self.pack(side = 'bottom', pady = 10)
        self.args = args

    def revert(self):
        for var, value in self.args:
            var.set(value)

class FileNamePanel(Panel):
    def __init__(self, parent, name_string, format_string):
        super().__init__(parent = parent)

        self.name_string = name_string
        self.name_string.trace('w', self.update_text)
        self.format_string = format_string

        ctk.CTkEntry(self, textvariable = self.name_string).pack(fill = 'x', padx = 20, pady = 5)
        frame = ctk.CTkFrame(self, fg_color = 'transparent')
        jpg_check = ctk.CTkCheckBox(frame, text='jpg', variable = self.format_string, command = lambda: self.click('jpg'),  onvalue='jpg', offvalue='png')
        png_check = ctk.CTkCheckBox(frame, text='png', variable = self.format_string, command = lambda: self.click('png'), onvalue='png', offvalue='jpg')

        jpg_check.pack(side = 'left', fill = 'x', expand = True)
        png_check.pack(side = 'right', fill = 'x', expand = True)
        frame.pack(expand = True, fill = 'x', padx = 20)

        self.output = ctk.CTkLabel(self, text = '')
        self.output.pack()

    def click(self, value):
        self.format_string.set(value)
        self.update_text()

    def update_text(self, *args):
        if self.name_string.get():
            text = self.name_string.get().replace(' ', '_') + '.' + self.format_string.get()
            self.output.configure(text = text)

class FilePathPanel(Panel):
    def __init__(self, parent, path_string):
        super().__init__(parent = parent)
        self.path_string = path_string

        ctk.CTkButton(self, text = 'Open Explorer', command = self.open_file_dialog).pack(pady = 5)
        ctk.CTkEntry(self, textvariable = self.path_string).pack(expand = True, fill = 'both', padx = 5, pady = 5)

    def open_file_dialog(self):
        self.path_string.set(filedialog.askdirectory())

class ImageChooser(Panel):
    def __init__(self, text, parent, path_string, func):
        super().__init__(parent=parent)
         
        self.path_string = path_string
        self.path_string.trace('w', self.update_text)
        
        ctk.CTkLabel(self, text=text).pack()
        ctk.CTkButton(self, text='Choose Image', command=self.open_file_dialog).pack(pady=5)
        
        self.output = ctk.CTkLabel(self, text='')
        self.output.pack()
        
        self.apply_button = ctk.CTkButton(self, text="Apply", command=func)
    
    def open_file_dialog(self):
        filetypes = (("jpeg, png, jpg, tif", "*.jpg *.png *.jpeg *.tif"), 
                     ("png files", "*.png"),
                     ("jpg files", "*.jpg"),
                     ("jpeg files", "*.jpeg"),
                     ("tif files", "*.tif")) 
        
        file_path = filedialog.askopenfilename(filetypes=filetypes)
        
        if file_path:
            self.path_string.set(file_path)
    
    def update_text(self, *args):
        file_path = self.path_string.get()
        
        if file_path:
            file_name = basename(file_path)
            self.output.configure(text=file_name)
            
            if file_name != "None":
                self.apply_button.pack()
            else:
                self.apply_button.pack_forget()
        else:
            self.apply_button.pack_forget()
            self.output.configure(text="")

    
    def update_text(self, *args):
        file_path = self.path_string.get()
        
        if file_path:
            file_name = basename(file_path)
            self.output.configure(text=file_name)
            
            if file_name != "None":
                self.apply_button.pack()
            else:
                self.apply_button.pack_forget()
        else:
            self.apply_button.pack_forget()
            self.output.configure(text="")

class ImageChooserWithDrop(Panel):
    def __init__(self, parent, path_string, options, func):
        super().__init__(parent=parent)
        
        self.path_string = path_string
        self.path_string.trace('w', self.update_text)
        
        self.main_func = func
        self.func = lambda:self.main_func(str(options[0]))
        self.options = options
        
        self.drop_chooser_string = ctk.StringVar(master=self, value='Histogram Matching')
        self.drop_chooser_string.trace('w', self.choose) 

        DropDownPanel(self, self.drop_chooser_string, options=options)
        ctk.CTkButton(self, text='Choose Image', command=self.open_file_dialog).pack(pady=5)
        
        self.output = ctk.CTkLabel(self, text='')
        self.output.pack()
        
        self.apply_button = ctk.CTkButton(self, text="Apply", command=self.func)
        
    def choose(self, *args):
        selected_option = self.drop_chooser_string.get()
        self.func = lambda: self.main_func(str(selected_option))
        self.apply_button.configure(command = self.func)
    
    def open_file_dialog(self):
        filetypes = (("jpeg, png, jpg, tif", "*.jpg *.png *.jpeg *.tif"), 
                     ("png files", "*.png"),
                     ("jpg files", "*.jpg"),
                     ("jpeg files", "*.jpeg"),
                     ("tif files", "*.tif")) 
        
        file_path = filedialog.askopenfilename(filetypes=filetypes)
        
        if file_path:
            self.path_string.set(file_path)
    
    def update_text(self, *args):
        file_path = self.path_string.get()
        
        if file_path:
            file_name = basename(file_path)
            self.output.configure(text=file_name)
            
            if file_name != "None":
                self.apply_button.pack()
            else:
                self.apply_button.pack_forget()
        else:
            self.apply_button.pack_forget()
            self.output.configure(text="")
        
class SaveButton(ctk.CTkButton):
    def __init__(self, parent, export_image, name_string, format_string, path_string):
        super().__init__(master = parent, text = 'Save', command = self.save)
        self.pack(side = 'bottom', pady = 10)
        self.export_image = export_image
        self.name_string = name_string
        self.format_string = format_string
        self.path_string = path_string

    def save(self):
        self.export_image(
                self.name_string.get(),
                self.format_string.get(),
                self.path_string.get())

