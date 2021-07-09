from tkinter import *
from tkinter import filedialog
from config import Config
from calculation import calculate
import cv2
import os
from PIL import Image, ImageTk

class yoloUI:
    def __init__(self):
        self.contentImage = None
        self.contentImageFilePath = ''
        self.cContent = None
        self.styleDirectoryPath = 'in/style'
        self.styleImage = None
        self.styleImageFilePath = ''
        self.cap = cv2.VideoCapture(0)
        self.root = Tk()
        #self.config = Config('vgg16')

        self.initializeRoot()
        self.initializeVideoCap()
        self.buildContentSelection()
        #self.buildNetworkSelection()

        #startStyleTransferButton = Button(self.root, text='Start Style Transfer', command= self.letsGo)

        self.buildButtonsWithStyleImages()

        self.show_frame()
        self.root.mainloop()

    def buildButtonsWithStyleImages(self):
        # buttons for styles
        fStyleButtons = Frame(self.root)
        fStyleButtons.grid(column=3, row=1)
        styleButtons = []
        for filename in os.listdir(self.styleDirectoryPath):
            filepath = os.path.join(self.styleDirectoryPath,filename)
            with Image.open(filepath) as img:
                img = img.resize((128,128))
                img = ImageTk.PhotoImage(img)
                styleButton = Button(fStyleButtons, text=filename, image=img, width=128, height=128, command= lambda : self.updateStyleImage(filepath))
                styleButton.image = img
                styleButtons.append(styleButton)

        bSelectOwnStyle = Button(fStyleButtons, text ="Select Own Style", command=lambda : self.getLocalFile(getContent=False))
        styleButtons.append(bSelectOwnStyle)

        # place buttons in list on canvas
        for button in styleButtons:
            button.grid()

    def initializeVideoCap(self):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    def buildNetworkSelection(self):
        fNetworkSelection = Frame(self.root)
        fNetworkSelection.grid(column=2)
        variable = StringVar(fNetworkSelection)
        variable.set("one")
        w = OptionMenu(fNetworkSelection, variable, "Resnet", "VGG")
        w.grid()

        contentWeightEntry = Entry(fNetworkSelection)#, textvariable = self.config.content_weight)
        styleWeightEntry = Entry(fNetworkSelection)#, textvariable = self.config.style_weight)
        tvWeightEntry = Entry(fNetworkSelection)#, textvariable = self.config.total_variation_weight)
        contentWeightEntry.grid()
        styleWeightEntry.grid()
        tvWeightEntry.grid()
        # saturationEntry = Entry(self.root, textvariable = self.config)

    def buildContentSelection(self):

        ## Frames on left side containing Webcam, Buttons, SelectedImage
        fContent = Frame(self.root)
        fContent.grid(row = 1, padx = 10, pady = 10)
        fContentButtons = Frame(self.root)
        fContentButtons.grid(row=2, pady = 10)
        fContentImage = Frame(self.root)
        fContentImage.grid(row=3)

        self.lmainVideo = Label(fContent)
        self.lmainVideo.pack(side = LEFT)

        bTakePhoto = Button(fContentButtons, text = "Take Photo", command=self.takePhoto)
        bSelectPhoto = Button(fContentButtons, text = "Select Image", command=self.getLocalFile)
        bSelectPhoto.grid(row= 1,column = 1, padx = 25)
        bTakePhoto.grid(row = 1,column = 2)

        self.cContent = Canvas(fContentImage, width= 640, height = 360,bg ='lightgray')
        self.cContent.grid(row = 1, column= 1)

    def initializeRoot(self):
        self.root.title('YOLO - Style Transfer')
        self.root.bind('<Escape>', lambda e: root.quit())

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
        img = img.resize((640,360))
        imgtk = ImageTk.PhotoImage(image=img)
        self.lmainVideo.imgtk = imgtk
        self.lmainVideo.configure(image=imgtk)
        self.lmainVideo.after(10, self.show_frame)

    def takePhoto(self):
        ret, frame = self.cap.read()
        cv2.imwrite('ContentIn.png', frame)
        self.updateContentImage('ContentIn.png')

    def letsGo(self):
        transfered_image = calculate(self.config, self.displayIntermediateResults())
        Image.imshow(transfered_image)

    def updateContentImage(self, newPath):
        self.contentImageFilePath = newPath
        self.cContent.delete("all")
        self.contentImage = Image.open(self.contentImageFilePath)
        self.contentImage = self.contentImage.resize((640,360))
        self.contentImage = ImageTk.PhotoImage(self.contentImage)

        self.cContent.create_image(0, 0, anchor=NW, image=self.contentImage)

    def updateStyleImage(self, newPath):
        self.styleImageFilePath = newPath
        self.styleImage = Image.open(self.styleImageFilePath)
    

def main():
    gui = yoloUI()

if __name__ == '__main__':
    main()
