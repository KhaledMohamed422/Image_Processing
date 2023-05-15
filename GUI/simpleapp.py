import tkinter as tk
# from tkinter import ttk
import ttkbootstrap as ttk

def convert():
    mile_input = (entryInt.get())
    km_output = mile_input * 1.61
    output_string.set(km_output)


# window
#window = ttk.Window(themename = 'journal')
window = ttk.Window(themename = 'darkly')
window.title('Demo')
window.geometry('300x150')

# title
title_label = ttk.Label(master = window, text = 'Miles to Kilometers', font = 'Calibri 22')
title_label.pack()

# input field
input_frame = ttk.Frame(master = window)
entryInt = tk.IntVar()
entry = ttk.Entry(master = input_frame, textvariable = entryInt)
button = ttk.Button(master = input_frame, text = 'Convert', command = convert)
entry.pack(side = 'left', padx = 10)
button.pack(side = 'left')
input_frame.pack(pady = 10)

#output
output_string = tk.StringVar()
output_label = ttk.Label(master = window, text = 'Output', textvariable = output_string)
output_label.pack(pady = 5)
# main loop to run
window.mainloop()

