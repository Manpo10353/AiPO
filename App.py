from tkinter import Tk, ttk, Canvas, Frame, filedialog
from PIL import Image, ImageTk
from skimage import color, filters, io
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import find_peaks
from skimage.morphology import disk


class App:
    def __init__(self, master):
        self.master = master
        master.title("Binaryzacja obrazu o wielomodalnym histogramie")
        master.geometry(f'{master.winfo_screenwidth()}x{master.winfo_screenheight()}+0+0')

        self.master.columnconfigure(0, weight=1)
        self.master.columnconfigure(1, weight=1)

        self.master.rowconfigure(1, weight=2)
        self.master.rowconfigure(2, weight=2)

        add_image_button = ttk.Button(master, text="Dodaj obraz", command=self.image)
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

        self.original_image_frame.grid(column=0, row=1, sticky="nsew")
        self.histograms_rgb_frame.grid(column=1, row=1, sticky="nsew")
        self.binarized_image_frame.grid(column=0, row=2, sticky="nsew")
        self.grayscale_histogram_frame.grid(column=1, row=2, sticky="nsew")

    def image(self):

        self.clear_frame()
        img = self.insert_image()



        if img.any():

            self.show_image(img, self.original_image_frame)
            img_gray = color.rgb2gray(img)
            histograms, hist_gray = self.calculate_histograms(img)
            histograms.append(hist_gray)

            self.plot_histograms_rgb_gray(histograms, self.histograms_rgb_frame)

            hist_array = np.array(hist_gray)

            peaks, _ = find_peaks(hist_array, distance=20, width=2)
            if len(peaks) == 0:
                peaks, _ = find_peaks(hist_array, distance=20, width=1)

            thresholds = self.calculate_thresholds(img_gray, min(len(peaks), 5))
            binarized = self.binarize(img_gray, thresholds)

            kernel = disk(2)
            opened_image = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, kernel)
            closed_opened_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)

            binarized = closed_opened_image

            binarized_hist, bins_gray = np.histogram(binarized, bins=256)
            binarized_hist = [binarized_hist[i] / (binarized.shape[0] * binarized.shape[1]) for i in range(256)]

            self.plot_histogram_thresholds(hist_gray, binarized_hist, thresholds, self.grayscale_histogram_frame)

            self.show_image(binarized, self.binarized_image_frame)

            self.save_image(binarized)

    @staticmethod
    def plot_histogram_thresholds(hist_gray, hist_bin, thresholds, frame):
        fig, ax1 = plt.subplots()

        ax1.plot(hist_bin, color='gray', label='Binarized hist')
        ax1.set_ylabel('Binarized Histogram', color='gray')
        ax1.tick_params('y', colors='gray')
        ax1.legend(loc='upper left')

        for threshold in thresholds:
            ax1.axvline(x=threshold, color='r', linestyle='--')

        ax2 = ax1.twinx()
        ax2.plot(hist_gray, color='r', label='hist_gray')
        ax2.set_ylabel('Grayscale Histogram', color='r')
        ax2.tick_params('y', colors='r')
        ax2.legend(loc='upper right')

        plt.title('Grayscale Histogram with thresholds')

        canvas_for_gray_histogram = FigureCanvasTkAgg(fig, master=frame)
        canvas_for_gray_histogram.draw()
        canvas_for_gray_histogram.get_tk_widget().pack(fill='both', expand=True)

    @staticmethod
    def plot_histograms_rgb_gray(histograms, frame):
        plt.figure()
        for i, color_name in enumerate(['Red', 'Green', 'Blue']):
            plt.subplot(2, 2, i + 1)
            plt.plot(histograms[i], color=color_name.lower())
            plt.title(f'{color_name} Histogram')

        plt.subplot(2, 2, 4)
        plt.plot(histograms[3], color='gray')
        plt.title('Grayscale Histogram')

        canvas_for_histograms = FigureCanvasTkAgg(plt.gcf(), master=frame)
        canvas_for_histograms.draw()
        canvas_for_histograms.get_tk_widget().pack(fill='both', expand=True)

    @staticmethod
    def show_image(img, frame):
        img_resized = Image.fromarray(img.astype(np.uint8))
        img_resized = img_resized.resize(
            (frame.winfo_width(),
             frame.winfo_height()),
            Image.LANCZOS)

        canvas_for_image = Canvas(frame)
        canvas_for_image.pack(fill='both', expand=True)

        canvas_for_image.image = ImageTk.PhotoImage(img_resized.resize((img_resized.width, img_resized.height)))
        canvas_for_image.create_image(0, 0, image=canvas_for_image.image, anchor='nw')

    @staticmethod
    def insert_image():
        file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files",
                                                                                    "*.png;*.jpg;*.jpeg;*.gif")])
        img = io.imread(file_path)
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        return img

    @staticmethod
    def save_image(img):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path and img is not None:
            pil_img = Image.fromarray(img)
            pil_img.save(file_path)

    def clear_frame(self):
        for frame in [self.original_image_frame, self.histograms_rgb_frame, self.binarized_image_frame,
                      self.grayscale_histogram_frame]:
            for widget in frame.winfo_children():
                widget.destroy()

    @staticmethod
    def calculate_thresholds(img_gray, peaks):
        optimal_thresholds = filters.threshold_multiotsu(img_gray, classes=peaks)
        return (optimal_thresholds * 255).astype(np.uint8)

    @staticmethod
    def binarize(img, thresholds):
        img = (img * 255).astype(np.uint8)
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
