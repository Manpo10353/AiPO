from tkinter import Tk, ttk, Canvas, Frame, filedialog, Label
from PIL import Image, ImageTk
from skimage import io, color
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class App:
    def __init__(self, master):
        self.master = master
        master.title("Binaryzacja obrazu o wielomodalnym histogramie")
        master.geometry(f'{master.winfo_screenwidth()}x{master.winfo_screenheight()}+0+0')

        self.master.columnconfigure(0, weight=1)
        self.master.columnconfigure(1, weight=1)
        self.master.rowconfigure(1, weight=2)
        self.master.rowconfigure(2, weight=2)

        add_image_button = ttk.Button(master, text="Dodaj obraz", command=self.insert_image)
        add_image_button.grid(column=0, row=0)

        close_app_button = ttk.Button(master, text="Zamknij", command=master.destroy)
        close_app_button.grid(column=1, row=0)

        self.original_image_frame = Frame(master, bg="gray")
        self.histograms_rgb_frame = Frame(master, bg="lightgray")
        self.binarized_image_frame = Frame(master, bg="lightgray")
        self.grayscale_histogram_frame = Frame(master, bg="gray")

        for frame in [self.original_image_frame, self.histograms_rgb_frame, self.binarized_image_frame,
                      self.grayscale_histogram_frame]:
            frame.pack_propagate(False)

        # Place frame widgets in the grid
        self.original_image_frame.grid(column=0, row=1, sticky="nsew")
        self.histograms_rgb_frame.grid(column=1, row=1, sticky="nsew")
        self.binarized_image_frame.grid(column=0, row=2, sticky="nsew")
        self.grayscale_histogram_frame.grid(column=1, row=2, sticky="nsew")

    def insert_image(self):
        file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files",
                                                                                    "*.png;*.jpg;*.jpeg;*.gif")])
        if file_path:
            self.clear_frame()
            img = Image.open(file_path)
            img_array = np.array(img)
            canvas_for_image = Canvas(self.original_image_frame, bg='green')
            canvas_for_image.pack(fill='both', expand=True)

            canvas_for_image.image = ImageTk.PhotoImage(img.resize((self.original_image_frame.winfo_width(), self.original_image_frame.winfo_height()), Image.LANCZOS))
            canvas_for_image.create_image(0, 0, image=canvas_for_image.image, anchor='nw')

            histograms, hist_gray = self.calculate_histograms(img_array)

            plt.figure()
            for i, color_name in enumerate(['Blue', 'Green', 'Red']):
                plt.subplot(2, 2, i + 1)
                plt.plot(histograms[i], color=color_name.lower())
                plt.title(f'{color_name} Histogram')

            plt.subplot(2, 2, 4)
            plt.plot(hist_gray, color='gray')
            plt.title('Grayscale Histogram')

            canvas_for_histograms = FigureCanvasTkAgg(plt.gcf(), master=self.histograms_rgb_frame)
            canvas_for_histograms.draw()
            canvas_for_histograms.get_tk_widget().pack(fill='both', expand=True)

    def clear_frame(self):
        for frame in [self.original_image_frame, self.histograms_rgb_frame, self.binarized_image_frame,
                  self.grayscale_histogram_frame]:
            for widget in frame.winfo_children():
                widget.destroy()


    def calculate_histograms(self, img):
        img_gray = color.rgb2gray(img)

        histograms = [cv2.calcHist([img], [i], None, [256], [0, 256]) for i in range(3)]
        hist_gray, bins_gray = np.histogram(img_gray.flatten(), bins=256, range=[0, 1])

        return histograms, hist_gray


if __name__ == "__main__":
    root = Tk()
    app = App(root)
    root.mainloop()