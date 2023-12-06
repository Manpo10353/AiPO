from App import App
from tkinter import Tk

# Create the root window (master)
root = Tk()

# Create an instance of the App class with the master as an argument
app = App(master=root)

# Start the Tkinter event loop
root.mainloop()