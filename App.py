import customtkinter
from tkinter import filedialog
from skimage import io, color
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class HistogramFrame(customtkinter.CTkFrame):
    def __init__(self, master, title):
        super().__init__(master)
        self.title = title
        self.histogram_frame = customtkinter.CTkFrame(self)
        self.histogram_frame.grid(row=0, column=0, padx=10, pady=(10, 10), sticky="nsew")

        self.histogram_canvas_1 = customtkinter.CTkCanvas(self.histogram_frame,
                                                         )
        self.histogram_canvas_2 = customtkinter.CTkCanvas(self.histogram_frame)
        self.histogram_canvas_3 = customtkinter.CTkCanvas(self.histogram_frame)
        self.histogram_canvas_4 = customtkinter.CTkCanvas(self.histogram_frame)

        self.histogram_canvas_1.pack(side="top", fill="both", expand=True)
        self.histogram_canvas_2.pack(side="top", fill="both", expand=True)
        self.histogram_canvas_3.pack(side="top", fill="both", expand=True)
        self.histogram_canvas_4.pack(side="top", fill="both", expand=True)

    def display_histogram(self, img):
        img_gray = color.rgb2gray(img)
        colors = ['Blue', 'Green', 'Red']

        histograms = [cv2.calcHist([img], [i], None, [256], [0, 256]) for i in range(3)]
        hist_gray, bins_gray = np.histogram(img_gray.flatten(), bins=256, range=[0, 1])

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()

        for i in range(3):
            axes[i].plot(histograms[i], color=colors[i])
            axes[i].set_title(colors[i])

        axes[3].plot(hist_gray, color='black')
        axes[3].set_title('Grayscale')

        canvas = FigureCanvasTkAgg(fig, master=self.histogram_canvas_1)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side="top", fill="both", expand=True)


class PhotoFrame(customtkinter.CTkFrame):
    def __init__(self, master, title):
        super().__init__(master)

        self.title = title
        self.photo_frame = customtkinter.CTkFrame(self)

        self.photo_frame.grid(row=0, column=0, padx=10, pady=(10, 10), sticky="nsew")
        self.after_idle(self.setup_canvas)

    def setup_canvas(self):
        # Ustawienie wymiarów CTkCanvas zgodnie z wymiarami self.photo_frame
        width = self.photo_frame.winfo_width()
        height = self.photo_frame.winfo_height()
        self.canvas = customtkinter.CTkCanvas(self.photo_frame, width=width, height=height)
        self.canvas.pack(side="top", fill="both", expand=True, anchor="center")

    def display_image(self, img):
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.axis("off")
        canvas = FigureCanvasTkAgg(fig, master=self.canvas)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side="top", fill="both", expand=True, anchor="center")


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("Binaryzacja obrazu wielomodalnego")

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f"{screen_width}x{screen_height}")

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        button = customtkinter.CTkButton(self, text="Dodaj zdjęcie", command=self.button_callback)
        button.grid(row=0, column=0, columnspan=2, padx=20, pady=20, sticky="nsew")

        self.photo_frame = PhotoFrame(self, "Zdjęcie")
        self.photo_frame.grid(row=1, column=0, padx=(10, 10), pady=(10, 10), sticky="nsew")

        self.histogram_frame = HistogramFrame(self, "Histogram")
        self.histogram_frame.grid(row=1, column=1, padx=(10, 10), pady=(10, 10), sticky="nsew")

    def button_callback(self):
        file_path = filedialog.askopenfilename(title="Wybierz plik", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            img = io.imread(file_path)
            self.photo_frame.display_image(img)
            self.histogram_frame.display_histogram(img)
