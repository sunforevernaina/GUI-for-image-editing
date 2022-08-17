#============= IMAGE PROCESSING : ASSIGNMENT_1 ==========================
# This is the private file demonstrating basic image operations using a GUI on different
# type of image i.e. Gray, RGB and HSV.

# Author : Sunaina Saxena
#========================================================================

#============= Libraries and Import Modules==============================
from tkinter import *
import numpy as np
from PIL import ImageTk,Image
from tkinter import filedialog
from tkinter import simpledialog
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import cv2
import pickle
import colorsys
#==========================================================================

#================ GLOBALS ===================================================

# Global variable that will always contain previously operated image
# that will be used further for next operation selected by user through GUI.
previous_image= ""

#============================================================================

def image_load():
# Routine Description : This routine bascially load the image in  image load area of GUI
#                       Operations can be further performed on this loaded image.
# Arguments :
#       NONE
# Return :
#       NONE
# Routine Trigger mechanism : GUI push button "load_image".


    # Global, where the updated image has to be stored.
    global previous_image
    # Global, to get original loaded image for undo all operation
    global original_image
    # Dialog box to browse image file, this will support all types of standard images
    # Extensions like jpg, png etc.
    root.filename=filedialog.askopenfilename(initialdir='at images',
                                             title='image',filetypes=(('JPG files','*.JPG'),('all files','*.*')))
    # Open the selected image and get the object handle
    user_image = ImageTk.PhotoImage(Image.open(root.filename))
    # We need to use canvas to customise our GUI.
    user_image_canvas =my_canvas.create_image(0,0,anchor=NW,image=user_image)
    # Updating the previous image with selected user image file object.
    previous_image=Image.open(root.filename)
    original_image=previous_image
    root.mainloop()
#======================== End of image_load =============================

def rgb_hsv_rgb():
# Routine Description : This routine will take user selcted RGB image and covert it into HSV
#                       and then further convert back to orignial RGB from HSV.
# Arguments :
#       NONE
# Return :
#       NONE
# Routine Trigger mechanism : GUI push button "rgb_hsv_rgb".

    # Global, where the updated image has to be stored.
    global previous_image
    # Since we are implementing UNDO operation so we need to update our pickle database file
    # with previous image before performing any operation so that we reload our old one whenever
    # user performed UNDO operation. Here we are opening the database handle to update the latest image
    f = open('image_database.pkl', 'wb')
    # Dumping current image into pickle file database.
    pickle.dump(previous_image,f)
    # Since we are done with operation, lets close the database handle.
    f.close()
    # Splitting RGB image to get raw values.
    r,g,b = previous_image.split()
    # Arrays to store the HSV component of the image repectively.
    Hdat = []
    Sdat = []
    Vdat = []
    # Iterating through RGB raw values and convert them to HSV component.
    for raw_data_r,raw_data_g,raw_data_b in zip(r.getdata(),g.getdata(),b.getdata()) :
        h,s,v = colorsys.rgb_to_hsv(raw_data_r/255.,raw_data_g/255.,raw_data_b/255.)
        Hdat.append(int(h*255.))
        Sdat.append(int(s*255.))
        Vdat.append(int(v*255.))
    # Updating RGB component with HSV component values respectively.
    r.putdata(Hdat)
    g.putdata(Sdat)
    b.putdata(Vdat)

    hsv_image =Image.merge('RGB',(r,g,b))
    hsv_image_array=np.array(hsv_image)
    # Convert HSV to RGB image array using openCV.
    rgb_image_array=cv2.cvtColor(hsv_image_array,cv2.COLOR_HSV2RGB)
    rgb_image=Image.fromarray(rgb_image_array)
    # Display the HSV and further converted RGB image.
    hsv_image.show('HSV')
    rgb_image.show('RGB')

#====================End of rgb_hsv_rgb ================================

def equalized():
# Routine Description : This routine will take loaded image converts it
# into single channel image(here converted into grayscale and then applied
# histogram equalization============================
#                       .
# Arguments :
#       NONE
# Return :
#       NONE
# Routine Trigger mechanism : GUI push button "equalized".

    # Global, where the updated image has to be stored.
    global previous_image
    # Since we are implementing UNDO operation so we need to update our pickle database file
    # with previous image before performing any operation so that we reload our old one whenever
    # user performed UNDO operation. Here we are opening the database handle to update the latest image
    f = open('image_database.pkl', 'wb')
    # Dumping current image into pickle file database.
    pickle.dump(previous_image,f)
    # Since we are done with operation, lets close the database handle.
    f.close()
    #converting loaded image into grayscale.
    imgray=previous_image.convert(mode='L')
    #making image a numpy array for further operations.
    img_array=np.array(imgray)
    #counting the no. of occurences of each element of an array.
    # after flattening it into one dimension
    histogram_array=np.bincount(img_array.flatten(),minlength=256)
    #summing all pixel values.
    num_pixels=np.sum(histogram_array)
    #averaging all pixel values by dividing it with the sum of all pixel values.
    histogram_array=histogram_array/num_pixels
    #computing cumulative sum of all the array elements.
    cdf_array=np.cumsum(histogram_array)
    #for getting lowest value in cdf_array.
    transform_map=np.floor(255 * cdf_array).astype(np.uint8)
    img_list=list(img_array.flatten())
    #transforming 1D array to 2D array for display purpose.
    eq_img_list=[transform_map[p] for p in img_list]
    eq_img_array=np.reshape(np.array(eq_img_list),img_array.shape)
    eq_img=Image.fromarray(eq_img_array)
    eq_img.show()
    #updating previous image for next operation.
    previous_image= eq_img


#===================code for gamma correction=================================
def gamma_correction():
# Routine Description : This routine takes RGB image and
# perform gamma correction on it..
# Arguments :
#       NONE
# Return :
#       NONE
# Routine Trigger mechanism : GUI push button "gamma_correct".

    # Global, where the updated image has to be stored.
    global previous_image
    # Since we are implementing UNDO operation so we need to update our pickle database file
    # with previous image before performing any operation so that we reload our old one whenever
    # user performed UNDO operation. Here we are opening the database handle to update the latest image
    f = open('image_database.pkl', 'wb')
    # Dumping current image into pickle file database.
    pickle.dump(previous_image,f)
    # Since we are done with operation, lets close the database handle.
    f.close()

    # Getting numpy image array of current loaded image
    image_array=np.array(previous_image)

    # Applying standard rule s = cr ^ Y, and converting them into gamma light and dark
    gamma_light = 255.0 * (image_array / 255.0) ** (1 / 2.2)
    gamma_dark = 255.0 * (image_array / 255.0) ** 2.2
    gamma_light_array = np.array(gamma_light, dtype = np.uint8)
    gamma_dark_array= np.array(gamma_dark,dtype=np.uint8)
    # Converting array into image for display.
    img=Image.fromarray(gamma_light_array)
    img.show()
    # Updating current image for further operations.
    previous_image = img

#===================code for log transform=======================================
def log_transform():
    # Routine Description : This routine takes RGB image and
    # perform log transform on it..
    # Arguments :
    #       NONE
    # Return :
    #       NONE
    # Routine Trigger mechanism : GUI push button "log_transform".

    # Global, where the updated image has to be stored.
    global previous_image
    # Since we are implementing UNDO operation so we need to update our pickle database file
    # with previous image before performing any operation so that we reload our old one whenever
    # user performed UNDO operation. Here we are opening the database handle to update the latest image
    f = open('image_database.pkl', 'wb')
    # Dumping current image into pickle file database.
    pickle.dump(previous_image,f)
    # Since we are done with operation, lets close the database handle.
    f.close()
    # Converting image into numpy array.
    image_array=np.array(previous_image)
    # Operation: S=C*log(1+r)
    c=255/np.log(1+255)
    log_img_array=c*np.log(1+image_array)
    log_image = np.array(log_img_array, dtype = np.uint8)
    # Converting array into image for display.
    img=Image.fromarray(log_image)
    img.show()
    # Updating current image for further operations.
    previous_image = img


#=============================code for negative of image===============================
def negative():
    # Routine Description : This routine takes RGB image and
    # give negative of it..
    # Arguments :
    #       NONE
    # Return :
    #       NONE
    # Routine Trigger mechanism : GUI push button "negative".

    # Global, where the updated image has to be stored.
    global previous_image
    # Since we are implementing UNDO operation so we need to update our pickle database file
    # with previous image before performing any operation so that we reload our old one whenever
    # user performed UNDO operation. Here we are opening the database handle to update the latest image
    f = open('image_database.pkl', 'wb')
    # Dumping current image into pickle file database.
    pickle.dump(previous_image,f)
    # Since we are done with operation, lets close the database handle.
    f.close()
    # Converting image to numpy array.
    img_array=np.array(previous_image)
    # Negating all the pixel values of image array.
    neg=255-img_array
    neg_image_array = np.array(neg, dtype = np.uint8)
    # neg= np.concatenate(( neg_array, negative), axis=1)
    # Converting array to image for display purpose
    neg_image=Image.fromarray(neg_image_array)
    neg_image.show()
    # Updating current image for further operations.
    previous_image = neg_image

#==============================code for undo=====================================
def undo():
# Routine Description : This routine will perform UNDO operation and reconstruct the
#                       last operational image using pickel module.
# Arguments :
#       NONE
# Return :
#       NONE
# Routine Trigger mechanism : GUI push button "undo".

    # Global variable to keep track of the last image.
    global previous_image
    # Since we were always storing our image handle into pickle database handle
    # All we need to do here to just pull the recent image object and update it with previous image.
    f = open('image_database.pkl', 'rb')
    image = pickle.load(f)
    image.show()
    f.close()
    previous_image= image

#=================functions required for kernel operations on image=================================
def multi_convolver(image, kernel, iterations):
# Routine Description : This routine will perform convolution on supplied channel with provided kernel
#
# Arguments :
#       image : Channel image at which convolution has to be performed.
#       kernel : Defined window for convolution.
#       iterations : The value which defines the number of times user wants to perform convolution.
# Return :
#       Convolved channel image.
# Routine Trigger mechanism : It just a supporting function can be used as subroutine.

    # Applying "iterations" times convolution on channel image with zero padding
    for i in range(iterations):
        image = convolve2d(image, kernel, 'same', boundary = 'fill',
                           fillvalue = 0)
    return image

def convolver_rgb(image, kernel, iterations = 1):
# Routine Description : This routine will perform convolution on supplied RGB image with provided kernel.
#                       It will return the convolved RGB image and also display channel convoled image separately
# Arguments :
#       image : Channel image at which convolution has to be performed.
#       kernel : Defines window for convolution.
#       iterations : The value which defines the number of times convolution user wants to perform.
# Return :
#       Convolved RGB image.
# Routine Trigger mechanism : It just a supporting function can be used as subroutine.

    # Applying convolution on each RGB image channel with supplied kernel.
    convolved_image_r = multi_convolver(image[:,:,0], kernel,
                                        iterations)
    convolved_image_g = multi_convolver(image[:,:,1], kernel,
                                        iterations)
    convolved_image_b  = multi_convolver(image[:,:,2], kernel,
                                         iterations)

    # Configuring the subplot for our image output
    fig, axis = plt.subplots(1,3, figsize = (17,10))

    # Displaying the channel convolved images.
    axis[0].imshow(abs(convolved_image_r), cmap='Reds')
    axis[0].set_title(f'Red', fontsize = 15)

    axis[1].imshow(abs(convolved_image_g), cmap='Greens')
    axis[1].set_title(f'Green', fontsize = 15)

    axis[2].imshow(abs(convolved_image_b), cmap='Blues')
    axis[2].set_title(f'Blue', fontsize = 15)

    # Constructing RGB convolved image using channel convolved images
    image[:,:,0] = convolved_image_r
    image[:,:,1] = convolved_image_g
    image[:,:,2] = convolved_image_b

    # Getting image object from RGB convolved image array.
    convolved_rgb = Image.fromarray(image)

    [axi.set_axis_off() for axi in axis.ravel()]

    return convolved_rgb

#===============================code for blurr operation on RGB images=================================
def average():
    # Routine Description : This routine takes RGB image and performs smoothing on it..
    # Arguments :
    #       NONE
    # Return :
    #       NONE
    # Routine Trigger mechanism : GUI push button "average_RGB".

    # Global, where the updated image has to be stored.
    global previous_image
    # Since we are implementing UNDO operation so we need to update our pickle database file
    # with previous image before performing any operation so that we reload our old one whenever
    # user performed UNDO operation. Here we are opening the database handle to update the latest image
    f = open('image_database.pkl', 'wb')
    # Dumping current image into pickle file database.
    pickle.dump(previous_image,f)
    # Since we are done with operation, lets close the database handle.
    f.close()

    # Defiition of the kernel which we are using for our convolution process.
    kernel = (1 / 16.0) * np.array([[1., 2., 1.],
                                    [2., 4., 2.],
                                    [1., 2., 1.]])

    # We are opening a input dialog box here to ask the depth of averaging that will define the
    # level at which we have to blur out image.
    depth_of_average = simpledialog.askinteger("Depth of Averaging", "Please Enter depth of averaging")
    # Converting image to numpy array.
    img_array = np.array(previous_image)
    # Convolving RGB image with defined kernel and user supplied depth of averaging.
    convolved_rgb_img = convolver_rgb(img_array, kernel, depth_of_average)
    con=np.array(convolved_rgb_img)
    # Display the output convolved image.
    convolved_rgb_img.show()
    # Updating our image with last performed operation.
    previous_image=convolved_rgb_img

#=======================================code for sharpening RGB image========================================
def sharpen():
    # Routine Description : This routine takes RGB image and performs sharpening on it..
    # Arguments :
    #       NONE
    # Return :
    #       NONE
    # Routine Trigger mechanism : GUI push button "sharpen_RGB".

    # Global, where the updated image has to be stored.
    global previous_image
    # Since we are implementing UNDO operation so we need to update our pickle database file
    # with previous image before performing any operation so that we reload our old one whenever
    # user performed UNDO operation. Here we are opening the database handle to update the latest image
    f = open('image_database.pkl', 'wb')
    # Dumping current image into pickle file database.
    pickle.dump(previous_image,f)
    # Since we are done with operation, lets close the database handle.
    f.close()

    # Defiition of the kernel which we are using for our convolution process.
    sharpen = np.array([[0, -1, 0],
                        [-1, 4, -1],
                        [0, -1, 0]])

    # We are opening a input dialog box here to ask the depth of sharpening that will define the
    # level at which we have to blur out image.
    depth_of_sharpening = simpledialog.askinteger("Depth of sharpening",
                                                  "Please Enter depth of sharpening)")
    # Converting image to numpy array.
    img_array = np.array(previous_image)
    # Convolving RGB image with defined kernel and user supplied depth of averaging.
    convolved_rgb_img = convolver_rgb(img_array, sharpen, depth_of_sharpening)
    # Display the output convolved image.
    convolved_rgb_img.show()
    # Updating our image with last performed operation.
    previous_image=convolved_rgb_img


#===============================code for edge_detection of RGB image=========================================
def edge_detect():
    global previous_image
    # Routine Description : This routine takes RGB image and performs edge detection on it..
    # Arguments :
    #       NONE
    # Return :
    #       NONE
    # Routine Trigger mechanism : GUI push button "edge_detect_RGB".

    # Global, where the updated image has to be stored.
    global previous_image
    # Since we are implementing UNDO operation so we need to update our pickle database file
    # with previous image before performing any operation so that we reload our old one whenever
    # user performed UNDO operation. Here we are opening the database handle to update the latest image
    f = open('image_database.pkl', 'wb')
    # Dumping current image into pickle file database.
    pickle.dump(previous_image,f)
    # Since we are done with operation, lets close the database handle.
    f.close()

    # Defiition of the kernel which we are using for our convolution process.
    edge_detect = np.array([[1., 0, -1.],
                        [2., 0, -2.],
                        [1.,0,-1.]])
    # Converting image to numpy array.
    img_array = np.array(previous_image)
    # Convolving RGB image with defined kernel and user supplied depth of averaging.
    convolved_rgb_img = convolver_rgb(img_array, edge_detect, 1)
    # Display the output convolved image.
    convolved_rgb_img.show()
    # Updating our image with last performed operation.
    previous_image=convolved_rgb_img


#===========================function for kernel operations on grayscale image==============================


def convolve2D(image, kernel, padding=0):
    # Routine Description : This routine will perform convolution on supplied grayscale image with provided kernel.
    #                       It will return the convolved image.
    # Arguments :
    #       image :  Image at which convolution has to be performed.
    #       kernel : Defines window for convolution.
    #       padding: Defines padding with zeroes.
    # Return :
    #       Convolved grayscale image.
    # Routine Trigger mechanism : It just a supporting function can be used as subroutine.

    # Cross Correlation by flipping the kernel
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]
    strides=1
    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output
#======================edge detection of GRAYSCALE images==================================
def edge_detect_gray():
    # Routine Description : This routine takes RGB image converts it into grayscale
    # and performs edge detection on it..
    # Arguments :
    #       NONE
    # Return :
    #       NONE
    # Routine Trigger mechanism : GUI push button "edge_detect_gray".

    # Global, where the updated image has to be stored.
    global previous_image
    # Since we are implementing UNDO operation so we need to update our pickle database file
    # with previous image before performing any operation so that we reload our old one whenever
    # user performed UNDO operation. Here we are opening the database handle to update the latest image
    f = open('image_database.pkl', 'wb')
    # Dumping current image into pickle file database.
    pickle.dump(previous_image,f)
    # Since we are done with operation, lets close the database handle.
    f.close()
    image=previous_image
    #converting to grayscale
    imgray=image.convert(mode='L')
    array_gray=np.array(imgray)
    # Definition of kernel for edge detection.
    edge_detect=np.array([[1.,2.,1.],
                            [0.,0.,0.],
                            [-1, -2, -1]])

    # Calling function for convolution with kernel.
    output= convolve2D(array_gray,edge_detect, padding=2)
    # Converting back from array to image for display purpose.
    edge=Image.fromarray(output)
    edge.show()
    # Updating image for next operation.
    previous_image=edge


#======================code for blurring of GRAYSCALE images==================================
def average_gray():
    # Routine Description : This routine takes RGB image converts it into grayscale
    # and performs smoothing on it..
    # Arguments :
    #       NONE
    # Return :
    #       NONE
    # Routine Trigger mechanism : GUI push button "average_gray".

    # Global, where the updated image has to be stored.
    global previous_image
    # Since we are implementing UNDO operation so we need to update our pickle database file
    # with previous image before performing any operation so that we reload our old one whenever
    # user performed UNDO operation. Here we are opening the database handle to update the latest image
    f = open('image_database.pkl', 'wb')
    # Dumping current image into pickle file database.
    pickle.dump(previous_image,f)
    # Since we are done with operation, lets close the database handle.
    f.close()
    image=previous_image
    # Converting RGB image to grayscale.
    imgray=image.convert(mode='L')
    array_gray=np.array(imgray)
    # definition of averaging kernel.
    average=(1/16)*np.array([[1.,1.,1.,1.],
                            [1.,1.,1.,1.],
                            [1., 1., 1.,1.],
                            [1.,1.,1.,1.]])

    # Calling function for convolution with kernel.
    output= convolve2D(array_gray,average, padding=2)
    # Converting array to image for display purpose.
    avg_image=Image.fromarray(output)
    avg_image.show()
    # Updating image to perform next operation.
    previous_image=avg_image


def sharp_gray():
    # Routine Description : This routine takes RGB image converts it into grayscale
    # and performs sharpening on it..
    # Arguments :
    #       NONE
    # Return :
    #       NONE
    # Routine Trigger mechanism : GUI push button "sharpen_gray".

    # Global, where the updated image has to be stored.
    global previous_image
    # Since we are implementing UNDO operation so we need to update our pickle database file
    # with previous image before performing any operation so that we reload our old one whenever
    # user performed UNDO operation. Here we are opening the database handle to update the latest image
    f = open('image_database.pkl', 'wb')
    # Dumping current image into pickle file database.
    pickle.dump(previous_image,f)
    # Since we are done with operation, lets close the database handle.
    f.close()
    image=previous_image
    # converting RGB image to grayscale
    imgray=image.convert(mode='L')
    gray_array=np.array(imgray)
    # Kernel definition for sharpening of image.
    sharp=np.array([[-1.,-1.,-1.],
                    [-1.,8.,-1.],
                    [-1., -1., -1.]])

    # Calling function for convolution operation.
    output= convolve2D(gray_array,sharp, padding=1)
    # Adding obtained sharp edges to original image to make original image sharper.
    sharp=output+gray_array
    # Converting from array to image for display.
    sharp_final=Image.fromarray(sharp)
    sharp_final.show()
    # Update current image for further operations.
    previous_image=sharp_final


def save_image():
    global previous_image
    previous_image.save('my_image1234.jpg')
    # Showing the saved image
    image1=Image.open('my_image1234.jpg')
    image1.show()

def undo_all():
    global original_image
    # for getting the original loaded image
    original_image.show()




#======================================Main program=============================================



root=Tk()
root.geometry('800x500')

#========================define background image=================================
img=ImageTk.PhotoImage(file='C:/Users/sunfo/IdeaProjects/GUI/at images/back.jpg')
#===========================create canvas====================================
my_canvas=Canvas(root,width=800,height=500)
my_canvas.pack(fill='both',expand=True)
#============================set image in canvas============================
my_canvas.create_image(0,0,image=img,anchor='nw')
#================================add a label======================================
my_canvas.create_text(400,50,text='IMAGE PROCESSING:)',font=50,fill='white')
#================================add some buttons================================
root.title('image processing assignment 1')
button1=Button(root,text='load_image',command=image_load)
button2=Button(root,text='gamma_correct',command=gamma_correction)
button3=Button(root,text='log_transform',command=log_transform)
button4=Button(root,text='equalized',command=equalized)
button5=Button(root,text='edge_detect_RGB',command=edge_detect)
button6=Button(root,text='average_RGB',command=average)
button7=Button(root,text='sharpen_RGB',command=sharpen)
button8=Button(root,text='negative',command=negative)
button10=Button(root,text='edge_detect_gray',command=edge_detect_gray)
button11=Button(root,text='average_gray',command=average_gray)
button12=Button(root,text='sharp_gray',command=sharp_gray)
button13=Button(root,text='rgb_hsv_rgb',command=rgb_hsv_rgb)
button9=Button(root,text='undo',command=undo)
button14=Button(root,text='save_current_image',command=save_image)
button15=Button(root,text='undo_all_changes',command=undo_all)
#============= Creating window for buttons in canvas===========================
button1_window=my_canvas.create_window(450,100,anchor='nw',window=button1)
button2_window=my_canvas.create_window(550,100,anchor='nw',window=button2)
button3_window=my_canvas.create_window(680,100,anchor='nw',window=button3)
button4_window=my_canvas.create_window(450,150,anchor='nw',window=button4)
button5_window=my_canvas.create_window(550,150,anchor='nw',window=button5)
button6_window=my_canvas.create_window(680,150,anchor='nw',window=button6)
button7_window=my_canvas.create_window(450,200,anchor='nw',window=button7)
button8_window=my_canvas.create_window(550,200,anchor='nw',window=button8)
button10_window=my_canvas.create_window(650,200,anchor='nw',window=button10)
button11_window=my_canvas.create_window(450,250,anchor='nw',window=button11)
button12_window=my_canvas.create_window(550,250,anchor='nw',window=button12)
button13_window=my_canvas.create_window(680,250,anchor='nw',window=button13)
button9_window=my_canvas.create_window(450,300,anchor='nw',window=button9)
button14_window=my_canvas.create_window(550,300,anchor='nw',window=button14)
button15_window=my_canvas.create_window(680,300,anchor='nw',window=button15)
root.mainloop()

