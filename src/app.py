from tkinter import *
from tkinter import filedialog
from config import Config
from calculation import calculate
import cv2
import os
from PIL import Image, ImageTk


class yoloUI:
    def __init__(self):
        self.content_image = None
        self.content_image_path = ''
        self.cContent = None
        folder, _ = os.path.split(os.path.realpath(__file__))
        self.style_images_path = os.path.join(folder, '../in/style')
        self.style_image = None
        self.style_image_path = ''
        self.cap = cv2.VideoCapture(0)
        self.root = Tk()
        self.config = Config('vgg16')

        self.initializeRoot()
        self.initializeVideoCap()
        self.build_content_selection()
        self.build_network_selection()

        self.build_style_selection()

        if not self.cap.isOpened():
            print("No Webcam Input detected")
        else:
            self.show_frame()
        self.root.mainloop()

    def build_style_selection(self):
        # buttons for styles
        style_buttons_frame = Frame(self.root)
        style_buttons_frame.grid(column=3, row=1)
        style_buttons = []
        for filename in os.listdir(self.style_images_path):
            filepath = os.path.join(self.style_images_path, filename)
            with Image.open(filepath) as img:
                img = img.resize((128, 128))
                img = ImageTk.PhotoImage(img)
                styleButton = Button(
                    style_buttons_frame, text=filename, image=img, width=128, height=128)
                styleButton.image = img
                styleButton.filepath = filepath
                styleButton.bind(
                    "<Button-1>", lambda e: self.updateStyleImage(e.widget.filepath))
                style_buttons.append(styleButton)

        bSelectOwnStyle = Button(style_buttons_frame, text="Select Own Style",
                                 command=lambda: self.getLocalFile(getContent=False))
        style_buttons.append(bSelectOwnStyle)

        # place buttons in list on canvas
        i = 0
        for button in style_buttons:
            button.grid(row=int((i/2)+1), column=int((i % 2)+1))
            i += 1

    def initializeVideoCap(self):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    def build_network_selection(self):
        network_selection_frame = Frame(self.root)
        network_selection_frame.grid(column=2, row=1)
        variable = StringVar(network_selection_frame)
        variable.set("one")

        w = OptionMenu(network_selection_frame, variable, "Resnet", "VGG")
        w.grid()

        self.content_weight_entry = Entry(network_selection_frame)
        self.content_weight_entry.insert(0, self.config.content_weight)
        self.content_weight_entry.grid()

        self.style_weight_entry = Entry(network_selection_frame)
        self.style_weight_entry.insert(0, self.config.style_weight)
        self.style_weight_entry.grid()

        self.tv_weight_entry = Entry(network_selection_frame)
        self.tv_weight_entry.insert(0, self.config.total_variation_weight)
        self.tv_weight_entry.grid()

        start_styleTransfer_button = Button(
            network_selection_frame, text='Start Style Transfer', command=self.start_style_transfer)
        start_styleTransfer_button.grid()

    def build_content_selection(self):

        # Frames on left side containing Webcam, Buttons, SelectedImage
        fLeft = Frame(self.root)
        fLeft.grid(row=1, column=1)
        content_selection_frame = Frame(
            fLeft, width=640, height=360, bg='lightgray')
        content_selection_frame.grid_propagate(0)
        content_selection_frame.grid(row=1, padx=10, pady=10)
        content_buttons_frame = Frame(fLeft)
        content_buttons_frame.grid(row=2, pady=10)
        content_image_frame = Frame(fLeft, pady=10, padx=10)
        content_image_frame.grid(row=3)

        self.lmainVideo = Label(content_selection_frame, bg='lightgray')
        self.lmainVideo.grid(row=1, column=1)

        take_photo_button = Button(
            content_buttons_frame, text="Take Photo", command=self.takePhoto)
        select_photo_button = Button(
            content_buttons_frame, text="Select Image", command=self.getLocalFile)
        select_photo_button.grid(row=1, column=1, padx=25)
        take_photo_button.grid(row=1, column=2)

        self.cContent = Canvas(content_image_frame,
                               width=640, height=360, bg='lightgray')
        self.cContent.grid(row=1, column=1)

    def initializeRoot(self):
        self.root.title('YOLO - Style Transfer')
        self.root.bind('<Escape>', lambda e: self.root.quit())

    def getLocalFile(self, getContent=True):
        filePath = filedialog.askopenfilename()
        if (getContent):
            self.updateContentImage(filePath)
        else:
            self.updateStyleImage(filePath)

    def show_frame(self):
        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        img = img.resize((640, 360))
        imgtk = ImageTk.PhotoImage(image=img)
        self.lmainVideo.imgtk = imgtk
        self.lmainVideo.configure(image=imgtk)
        self.lmainVideo.after(10, self.show_frame)

    def takePhoto(self):
        _, frame = self.cap.read()
        cv2.imwrite('ContentIn.png', frame)
        self.updateContentImage('ContentIn.png')

    def start_style_transfer(self):
        print(self.content_weight_entry.get(),
              self.style_weight_entry.get(), self.tv_weight_entry.get())
        self.config.content_weight = float(self.content_weight_entry.get())
        self.config.style_weight = float(self.style_weight_entry.get())
        self.config.total_variation_weight = float(self.tv_weight_entry.get())

        transfered_image = calculate(self.config, lambda x, y: 0)
        transfered_image.show()

    def updateContentImage(self, newPath):
        self.content_image_path = newPath
        self.cContent.delete("all")
        self.content_image = Image.open(self.content_image_path)
        self.config.set_content(self.content_image)
        self.content_image = self.content_image.resize((640, 360))
        self.content_image = ImageTk.PhotoImage(self.content_image)

        self.cContent.create_image(0, 0, anchor=NW, image=self.content_image)

    def updateStyleImage(self, newPath):
        self.style_image_path = newPath
        self.style_image = Image.open(self.style_image_path)
        self.config.set_style(self.style_image)


def main():
    gui = yoloUI()


if __name__ == '__main__':
    main()
