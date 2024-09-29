#! .venv\Scripts\python.exe

import tkinter as tk
import json
from tkinter import messagebox, ttk


class Editor:
    def __init__(self, config_file):
        self.config_file = config_file

        self.root = tk.Tk()
        self.root.title("Config Editor")
        self.root.geometry("800x800")
        self.root.configure(bg="#d9d9d9")

        self.config_data = self.load_config()
        self.entry_widgets = {}

        self.create_scrollable_area()

    def main_loop(self):
        self.root.mainloop()

    def create_scrollable_area(self):
        # Create a frame for the canvas and scrollbar
        self.frame = tk.Frame(self.root, bg="#d9d9d9")
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Create a canvas
        self.canvas = tk.Canvas(self.frame, bg="#d9d9d9")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add a scrollbar to the canvas
        self.scrollbar = ttk.Scrollbar(
            self.frame, orient=tk.VERTICAL, command=self.canvas.yview
        )
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Configure the canvas with the scrollbar
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        # Create another frame inside the canvas for your widgets
        self.inner_frame = tk.Frame(self.canvas, bg="#d9d9d9")
        self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")

        # Create widgets inside the scrollable frame
        self.create_config_editor()

    def style_widget(self, widget, fg_color, bg_color, font_size=12):
        widget.configure(
            fg=fg_color,
            bg=bg_color,
            font=("Arial", font_size),
            relief="raised",
        )

    def load_config(self):
        """Loads the configuration from the JSON file."""
        try:
            with open(self.config_file, "r") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            messagebox.showerror("Error", f"Failed to load config: {e}")
            return {}

    def save_config(self):
        """Saves the configuration to the JSON file."""
        try:
            with open(self.config_file, "w") as file:
                json.dump(self.config_data, file, indent=4)
            messagebox.showinfo("Success", "Configuration saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config: {e}")

    def create_config_editor(self):
        """Creates a dynamic form for editing configuration settings."""
        row = 0
        row = self.create_widgets_for_config(self.config_data, row, parent_key="")

        # Save button
        save_button = tk.Button(
            self.inner_frame, text="Save Config", command=self.save_config_changes
        )
        save_button.grid(row=row + 1, column=0, columnspan=2, padx=10, pady=20)

    def create_widgets_for_config(self, config, row, parent_key=""):
        """Recursively creates input fields for each key-value pair in the config."""
        for key, value in config.items():
            full_key = (
                f"{parent_key}.{key}" if parent_key else key
            )  # Construct key path

            if isinstance(value, dict):
                # If the value is a nested dictionary, create a label and recursively process it
                tk.Label(self.inner_frame, text=key, font=("Arial", 12, "bold")).grid(
                    row=row, column=0, padx=10, pady=10, columnspan=2
                )
                row += 1
                row = self.create_widgets_for_config(
                    value, row, full_key
                )  # Recurse into the nested dictionary
            else:
                # Otherwise, create an entry field for the key-value pair
                tk.Label(self.inner_frame, text=key, font=("Arial", 10)).grid(
                    row=row, column=0, padx=10, pady=10
                )
                entry = tk.Entry(self.inner_frame, font=("Arial", 8), width=40)
                entry.insert(0, str(value))  # Insert the current value as default text
                entry.grid(row=row, column=1, padx=10, pady=10, sticky="w")
                self.entry_widgets[full_key] = (
                    entry  # Store the reference to this entry widget
                )
                row += 1

        return row

    def update_config(self, config, parent_key=""):
        """Recursively updates the config data from the entry widgets."""
        for key, value in config.items():
            full_key = f"{parent_key}.{key}" if parent_key else key

            if isinstance(value, dict):
                # If the value is a nested dictionary, recurse into it
                self.update_config(value, full_key)
            else:
                # Otherwise, update the config value from the corresponding Entry widget
                if full_key in self.entry_widgets:
                    new_value = self.entry_widgets[full_key].get()
                    if isinstance(value, int):
                        config[key] = int(new_value)
                    elif isinstance(value, float):
                        config[key] = float(new_value)
                    else:
                        config[key] = new_value

    def save_config_changes(self):
        """Updates the config_data dictionary and saves the changes to the file."""
        # Update the config_data based on user input
        self.update_config(self.config_data)

        # Save updated config to file
        self.save_config()


def main():
    editor = Editor("config.json")  # Specify your config file name
    editor.main_loop()


if __name__ == "__main__":
    main()
