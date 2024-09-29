#! .venv\Scripts\python.exe
import tkinter as tk

class FactoryManager:
    def __init__(self):

        self.root = tk.Tk()
        self.root.title("Script Manager")
        self.root.geometry("800x800")
        # Set background color for the entire window
        self.root.configure(bg="#d9d9d9")

        self.button_width = 40  # Adjust button width as needed
        self.button_height = 2  # Adjust button height as needed
