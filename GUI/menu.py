import customtkinter as ctk
from panels import *

class Menu(ctk.CTkTabview):
    def __init__(self, parent, tab1_vars, tab2_vars, tab3_vars,tab4_vars, process_func, export_image):
        super().__init__(master = parent)
        self.grid(row = 0, column = 0, sticky = 'nsew', pady = 10, padx = 10)

        #tabs 
        self.add('Tab1')
        self.add('Tab2')
        self.add('Tab3')
        self.add('Tab4')
        self.add('Export')
        
        # widgets
        Tab1Frame(self.tab('Tab1'), tab1_vars, process_func)
        Tab2Frame(self.tab('Tab2'), tab2_vars, process_func)
        Tab3Frame(self.tab('Tab3'), tab3_vars, process_func)
        Tab4Frame(self.tab('Tab4'), tab4_vars, process_func)
        ExportFrame(self.tab('Export'), export_image)

class Tab1Frame(ctk.CTkFrame):
    def __init__(self, parent, tab_vars, process_func):
        super().__init__(master=parent, fg_color='transparent')
        self.pack(expand=True, fill='both')
        
        SwitchPanel(self, (tab_vars['grayscale'], 'B/W'), (tab_vars['invert'], 'invert'))
        SwitchPanel(self, (tab_vars['histogram_equalization'], 'Histogram Equalization'))
        SliderPanel(self, 'Brightness Offset', tab_vars['brightness'], 0, 100, lambda: process_func('Brightness'))
        SliderPanel(self, 'Power Low gamma', tab_vars['gamma'], 0, 5, lambda: process_func('power_trans')) 
        ImageChooserWithDrop(self, tab_vars['image_path'], tab_vars['drop_options'], process_func)

class Tab2Frame(ctk.CTkFrame):
    def __init__(self, parent, tab_vars, process_func):
        super().__init__(master = parent, fg_color = 'transparent')
        self.pack(expand = True, fill = 'both')
        SwitchPanel(self, (tab_vars['Draw_Hist'], 'Draw Histogram'))
        TwoEntryPanel(self, "Contrast Stretching", "New Min", "New Max", tab_vars['new_min'], tab_vars['new_max'], lambda:process_func('contrast streching'))
       

class Tab3Frame(ctk.CTkFrame):
    def __init__(self, parent, tab_vars, process_func):
        super().__init__(master = parent, fg_color = 'transparent')
        self.pack(expand = True, fill = 'both')
        
        # DropDownPanel(self, tab_vars['effect'], EFFECT_OPTIONS)
        SwitchPanel(self, (tab_vars['edge_det'], 'Edge'),( (tab_vars['Sharp'], 'Sharp')))
        SliderPanel_filter(self, 'Smoothing Filter', tab_vars['blur'],1,7,lambda: process_func('blur'),step=3)
        SliderPanel(self, 'Reduce Size', tab_vars['reduce'],1,7,lambda: process_func('reduce'))

        # SliderPanel(self, 'Contrast', tab_vars['contrast'], 1, 10, process_func)
class Tab4Frame(ctk.CTkFrame):
    def __init__(self, parent, tab_vars, process_func):
        super().__init__(master = parent, fg_color = 'transparent')
        self.pack(expand = True, fill = 'both')
        
        # DropDownPanel(self, tab_vars['effect'], EFFECT_OPTIONS)
        SwitchPanel(self, (tab_vars['High/Low Pass'], 'High/Low Pass'))
        OneEntryPanel(self, "Order of Butterworth Filter", "N: ", tab_vars["order"])
        SliderPanel(self, 'Ideal Filter', tab_vars['Ideal'],0,30,lambda: process_func('Ideal'))
        SliderPanel(self, 'Butterworth Filter', tab_vars['Butterworth'],0,50,lambda: process_func('Butterworth'))
        SliderPanel(self, 'Gaussian Filter', tab_vars['Gaussian'],0,50,lambda: process_func('Gaussian'))
class ExportFrame(ctk.CTkFrame):
    def __init__(self, parent, export_image):
        super().__init__(master = parent, fg_color = 'transparent')
        self.pack(expand = True, fill = 'both')

        self.name_string = ctk.StringVar()
        self.format_string = ctk.StringVar(value = 'jpg')
        self.path_string = ctk.StringVar()

        FileNamePanel(self, self.name_string, self.format_string)
        FilePathPanel(self, self.path_string)
        SaveButton(self, export_image, self.name_string, self.format_string, self.path_string)
