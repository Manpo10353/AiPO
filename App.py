from tkinter import Tk, ttk, Canvas, Frame, filedialog
from PIL import Image, ImageTk
from skimage import color, filters
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import find_peaks


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
        file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files","*.png;*.jpg;*.jpeg;*.gif")])
        if file_path:
            self.clear_frame()
            img = Image.open(file_path)
            img_array = np.array(img)
            img_gray = color.rgb2gray(img_array)
            canvas_for_image = Canvas(self.original_image_frame, bg='green')
            canvas_for_image.pack(fill='both', expand=True)

            canvas_for_image.image = ImageTk.PhotoImage(img.resize((self.original_image_frame.winfo_width(), self.original_image_frame.winfo_height()), Image.LANCZOS))
            canvas_for_image.create_image(0, 0, image=canvas_for_image.image, anchor='nw')

            histograms, hist_gray = self.calculate_histograms(img_array)

            plt.figure()
            for i, color_name in enumerate(['Red', 'Green', 'Blue']):
                plt.subplot(2, 2, i + 1)
                plt.plot(histograms[i], color=color_name.lower())
                plt.title(f'{color_name} Histogram')

            plt.subplot(2, 2, 4)
            plt.plot(hist_gray, color='gray')
            plt.title('Grayscale Histogram')

            canvas_for_histograms = FigureCanvasTkAgg(plt.gcf(), master=self.histograms_rgb_frame)
            canvas_for_histograms.draw()
            canvas_for_histograms.get_tk_widget().pack(fill='both', expand=True)

            derivative = np.diff(hist_gray)

            # Znajdź punkty zwrotne w pochodnej (zmiana kierunku)
            inflection_points = np.where(np.diff(np.sign(derivative)))[0]

            peaks, _ = find_peaks(hist_gray, distance=40)

            print(inflection_points, peaks)
            thresholds = self.calculate_thresholds(img_gray, peaks)

            binarized = self.binarize(img_gray, thresholds)
            binarized_hist, bins_gray = np.histogram(binarized, bins=256)
            binarized_hist = [binarized_hist[i] / (binarized.shape[0]*binarized.shape[1]) for i in range(256)]

            # Utwórz wykres
            fig, ax1 = plt.subplots()

            # Rysuj binarized_hist na pierwszej osi Y (lewa oś Y)
            ax1.plot(binarized_hist, color='gray', label='Binarized hist')
            ax1.set_ylabel('Binarized Histogram', color='gray')
            ax1.tick_params('y', colors='gray')
            ax1.legend(loc='upper left')

            # Dodaj linie pionowe dla progów
            for threshold in thresholds:
                ax1.axvline(x=threshold, color='r', linestyle='--')

            # Utwórz drugą oś Y (prawa oś Y)
            ax2 = ax1.twinx()
            ax2.plot(hist_gray, color='r', label='hist_gray')
            ax2.set_ylabel('Grayscale Histogram', color='r')
            ax2.tick_params('y', colors='r')
            ax2.legend(loc='upper right')

            plt.title('Grayscale Histogram with thresholds')

            # Utwórz widget Canvas Tkinter
            canvas_for_gray_histogram = FigureCanvasTkAgg(fig, master=self.grayscale_histogram_frame)
            canvas_for_gray_histogram.draw()
            canvas_for_gray_histogram.get_tk_widget().pack(fill='both', expand=True)

            canvas_for_image = Canvas(self.binarized_image_frame, bg='green')
            canvas_for_image.pack(fill='both', expand=True)
            binarized_image_pil = Image.fromarray(binarized)
            binarized_image_tk = ImageTk.PhotoImage(binarized_image_pil.resize((self.original_image_frame.winfo_width(), self.original_image_frame.winfo_height()), Image.LANCZOS))
            canvas_for_image.create_image(0, 0, image=binarized_image_tk, anchor='nw')

            canvas_for_image.image = binarized_image_tk


    def save_image(self, binarized_img):
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg")])
        if file_path:
            binarized_img.save(file_path)

    def clear_frame(self):
        for frame in [self.original_image_frame, self.histograms_rgb_frame, self.binarized_image_frame,self.grayscale_histogram_frame]:
            for widget in frame.winfo_children():
                widget.destroy()

    @staticmethod
    def calculate_thresholds(img_gray, peaks):
        optimal_thresholds = filters.threshold_multiotsu(img_gray, classes=len(peaks))
        return (optimal_thresholds*255).astype(np.uint8)

    @staticmethod
    def binarize(img, thresholds):
        img = (img*255).astype(np.uint8)
        gray = (np.linspace(0, 255, len(thresholds) + 1).astype(np.uint8))
        binarized = np.digitize(img, bins=thresholds).astype(np.uint8)
        binarized = np.choose(binarized, gray)
        return binarized

    @staticmethod
    def calculate_histograms(img):
        img_gray = color.rgb2gray(img)
        width, height = img_gray.shape
        num_pixels = width * height

        histograms = [cv2.calcHist([img], [i], None, [256], [0, 256]) for i in range(3)]
        hist_gray, bins_gray = np.histogram(img_gray.flatten(), bins=256, range=[0, 1])
        histograms = [histogram / num_pixels for histogram in histograms]
        hist_gray = [hist_gray[i] / num_pixels for i in range(256)]
        return histograms, hist_gray


if __name__ == "__main__":
    root = Tk()
    app = App(root)
    root.mainloop()
