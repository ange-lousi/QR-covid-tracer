from matplotlib import pyplot
from matplotlib.patches import Rectangle
from pyzbar.pyzbar import decode
import imageIO.png
import math
import statistics
from numpy import asarray


class Queue:
        def __init__(self):
            self.items = []

        def isEmpty(self):
            return self.items == []

        def enqueue(self, item):
            self.items.insert(0,item)

        def dequeue(self):
            return self.items.pop()

        def size(self):
            return len(self.items)


def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):
    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array

# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):
    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)

# This method packs together three individual pixel arrays for r, g and b values into a single array that is fit for
# use in matplotlib's imshow method
def prepareRGBImageForImshowFromIndividualArrays(r,g,b,w,h):
    rgbImage = []
    for y in range(h):
        row = []
        for x in range(w):
            triple = []
            triple.append(r[y][x])
            triple.append(g[y][x])
            triple.append(b[y][x])
            row.append(triple)
        rgbImage.append(row)
    return rgbImage
    
# This method takes a greyscale pixel array and writes it into a png file
def writeGreyscalePixelArraytoPNG(output_filename, pixel_array, image_width, image_height):
    # now write the pixel array as a greyscale png
    file = open(output_filename, 'wb')  # binary mode is important
    writer = imageIO.png.Writer(image_width, image_height, greyscale=True)
    writer.write(file, pixel_array)
    file.close()

def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for j in range(image_width):
            greyscale_pixel_array[i][j] = int(round((pixel_array_r[i][j] * 0.299) + (pixel_array_g[i][j] * 0.587) + (0.114* pixel_array_b[i][j])))
    greyArray = greyscale_pixel_array.copy()
    scaledGreyArray = contrastStretch(greyArray, image_width, image_height)
    return greyscale_pixel_array

def computeMinAndMaxValues(pixel_array, image_width, image_height):
    fmax = [max(p) for p in pixel_array]
    fhigh = max(fmax)
    fmin = [min(q) for q in pixel_array]
    flow = min(fmin)
    return flow, fhigh

def computeVerticalEdgesSobel(pixel_array, image_width, image_height):
    vertical = createInitializedGreyscalePixelArray(image_width, image_height)
    sobel_x = [[-1, 0, 1], 
               [-2, 0, 2], 
               [-1, 0, 1]]

    for x in range(1,image_height-1):
        for y in range(1,image_width-1):
            vertical[x][y] = (float((sobel_x[0][0] * pixel_array[x-1][y-1]) + (sobel_x[0][1] * pixel_array[x-1][y]) + (sobel_x[0][2] * pixel_array[x-1][y+1]) +
                                    (sobel_x[1][0] * pixel_array[x][y-1])                                           + (sobel_x[1][2] * pixel_array[x][y+1])   +
                                    (sobel_x[2][0] * pixel_array[x+1][y-1]) + (sobel_x[2][1] * pixel_array[x+1][y]) + (sobel_x[2][2] * pixel_array[x+1][y+1])))
    return vertical

def computeHorizontalEdgesSobel(pixel_array, image_width, image_height):
    horizontal = createInitializedGreyscalePixelArray(image_width, image_height)
    sobel_y = [ [-1, -2, -1], 
                [0,  0,  0], 
                [1,  2,  1]]
    
    for x in range(1,image_height-1):
        for y in range(1,image_width-1):
            horizontal[x][y] = (float((sobel_y[0][0] * pixel_array[x-1][y-1]) + (sobel_y[0][1] * pixel_array[x-1][y]) + (sobel_y[0][2] * pixel_array[x-1][y+1]) +
                                   (sobel_y[1][0] * pixel_array[x][y-1])   + (sobel_y[1][2] * pixel_array[x][y+1]) +
                                   (sobel_y[2][0] * pixel_array[x+1][y-1]) + (sobel_y[2][1] * pixel_array[x+1][y]) + (sobel_y[2][2] * pixel_array[x+1][y+1]))) 
    return horizontal

def edgeMagnitude(verticleSobelEdge, horizontalSobelEdge, image_width, image_height):
    gradient = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for j in range(image_width):
            gradient[i][j] = math.sqrt((pow(verticleSobelEdge[i][j], 2) + pow(horizontalSobelEdge[i][j], 2)))
    return gradient

def contrastStretch(array, image_width, image_height):
    scaledArray = createInitializedGreyscalePixelArray(image_width, image_height)
    theArray = array.copy()
    
    minMax = computeMinAndMaxValues(theArray, image_width, image_height)

    gmax = 255
    gmin = 0
    for i in range(0,image_height):
        for j in range(0,image_width):
                        
            if theArray[i][j] < 0:
                scaledArray[i][j] = 0
            
            if theArray[i][j] > 255:
                scaledArray[i][j] = 255
                
            if theArray[i][j] <= 255 and theArray[i][j] >= 0:
                if min(minMax) - max(minMax) != 0:
                    scaledArray[i][j] = round((theArray[i][j] - min(minMax)) * ((gmax - gmin)/(max(minMax) - min(minMax))))
    return scaledArray

def computeBoxAveraging3x3(pixel_array, image_width, image_height):
    BoxAverage_array = createInitializedGreyscalePixelArray(image_width, image_height)

    for y in range(1, image_height - 1):
        for x in range(1, image_width - 1):
            top_row = pixel_array[y - 1][x - 1] + pixel_array[y - 1][x] + pixel_array[y - 1][x + 1]
            middle_row = pixel_array[y][x - 1] + pixel_array[y][x] + pixel_array[y][x + 1]
            bottom_row = pixel_array[y + 1][x - 1] + pixel_array[y + 1][x] + pixel_array[y + 1][x + 1]
            BoxAverage_array[y][x] = (top_row + middle_row + bottom_row) / 9

    return BoxAverage_array

def convertBinary(pixel_array, threshHold, image_width, image_height):
    binaryArray = createInitializedGreyscalePixelArray(image_width, image_height);
    for i in range(image_height):
        for j in range(image_width):
            if  pixel_array[i][j] < threshHold:
                binaryArray[i][j] = 0
            else:
                binaryArray[i][j] = 255
    return binaryArray

def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    Labeled_img = createInitializedGreyscalePixelArray(image_width, image_height)
    Visited = createInitializedGreyscalePixelArray(image_width, image_height)
    Dictionary = {}

    Current_label = 1

    for y in range(image_height):
        for x in range(image_width):

            if (pixel_array[y][x] != 0) and (Visited[y][x] == 0):
                q = Queue()
                q.enqueue([y, x])
                count = 0
                while q.size() != 0:
                    n = q.dequeue()
                    Labeled_img[n[0]][n[1]] = Current_label

                    if (n[1] - 1 >= 0) and (pixel_array[n[0]][n[1] - 1] != 0) and (Visited[n[0]][n[1] - 1] == 0):
                        q.enqueue([n[0], n[1] - 1])
                        Visited[n[0]][n[1] - 1] = 1
                        Labeled_img[n[0]][n[1] - 1] = Current_label
                        count += 1

                    if ((n[1] + 1 < image_width) and (pixel_array[n[0]][n[1] + 1] != 0) and (
                            Visited[n[0]][n[1] + 1] == 0)):
                        q.enqueue([n[0], n[1] + 1])
                        Visited[n[0]][n[1] + 1] = 1
                        Labeled_img[n[0]][n[1] + 1] = Current_label
                        count += 1

                    if (n[0] - 1 >= 0) and (pixel_array[n[0] - 1][n[1]] != 0) and (Visited[n[0] - 1][n[1]] == 0):
                        q.enqueue([n[0] - 1, n[1]])
                        Visited[n[0] - 1][n[1]] = 1
                        Labeled_img[n[0] - 1][n[1]] = Current_label
                        count += 1

                    if (n[0] + 1 < image_height) and (pixel_array[n[0] + 1][n[1]] != 0) and (Visited[n[0] + 1][n[1]] == 0):
                        q.enqueue([n[0] + 1, n[1]])
                        Visited[n[0] + 1][n[1]] = 1
                        Labeled_img[n[0] + 1][n[1]] = Current_label
                        count += 1

                Dictionary.update({Current_label: count})
                Current_label += 1

    max_key = max(Dictionary, key=Dictionary.get)
    component_analysis_array = createInitializedGreyscalePixelArray(image_width, image_height)

    for y in range(image_height):
        for x in range(image_width):
            if Labeled_img[y][x] == max_key:
                component_analysis_array[y][x] = 255
            else:
                component_analysis_array[y][x] = 0

    return component_analysis_array


def boxCoordinates(pixel_array, image_width, image_height):
    min_x = image_width
    min_y = image_height
    max_x = 0
    max_y = 0
    
    for y in range(image_height):
        for x in range(image_width):
            if pixel_array[y][x] == 255:
                if x < min_x:
                    min_x = x
                if y < min_y:
                    min_y = y
                if x > max_x:
                    max_x = x
                if y > max_y:
                    max_y = y

    return (min_x, min_y, max_x, max_y)

def main():
    filename = "./images/covid19QRCode/poster1small.png"

    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(filename)
    
    #STEP 1
    greyRGBimage = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    scaledGreyRGBimage = contrastStretch(greyRGBimage, image_width, image_height)
    #pyplot.imshow(scaledGreyRGBimage, cmap="gray")
    pixel_array = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)


    #STEP 2
    gy = computeHorizontalEdgesSobel(scaledGreyRGBimage, image_width, image_height)
    #pyplot.imshow(gy, cmap="gray")

    #STEP 3
    gx = computeVerticalEdgesSobel(scaledGreyRGBimage, image_width, image_height)
    #pyplot.imshow(gx, cmap="gray")

    #STEP 4
    gradientMagnitude = edgeMagnitude(gy, gx, image_width, image_height)
    #pyplot.imshow(gradientMagnitude, cmap="gray")

    #STEP 5
    meanSmooth = []
    for i in range(10):
        if i == 0:
            meanSmooth = computeBoxAveraging3x3(gradientMagnitude, image_width, image_height)
        else:
            meanSmooth = computeBoxAveraging3x3(meanSmooth, image_width, image_height)
    meanSmoothStretched = contrastStretch(meanSmooth, image_width, image_height)
    #pyplot.imshow(meanSmooth, cmap="gray")

    #STEP 6
    threshold_value = 70
    binary = convertBinary(meanSmoothStretched, threshold_value, image_width, image_height)
    #pyplot.imshow(binary, cmap="gray")

    #STEP 8
    largestConnectedComponent = computeConnectedComponentLabeling(binary, image_width, image_height)
    #pyplot.imshow(largestConnectedComponent, cmap="gray")

    #STEP 9
    (min_x, min_y, max_x, max_y) = boxCoordinates(largestConnectedComponent, image_width, image_height)
    
    x_cord = min_x
    y_cord = min_y 
    width = max_x - min_x
    height = max_y - min_y
    pyplot.imshow(prepareRGBImageForImshowFromIndividualArrays(px_array_r, px_array_g, px_array_b, image_width, image_height))
    
    axes = pyplot.gca()
    # create a width x height rectangle that starts at location x_cord, y_cord, with a line width of 3
    rectangle = Rectangle((x_cord, y_cord), width, height, linewidth=3, edgecolor='g', facecolor='none')

    axes.add_patch(rectangle)

    # Extension - QR Code Decoder
    qrCode_array = createInitializedGreyscalePixelArray(width, height)
    for y in range(height):  # These for loops cut out the QR Code from the pixel_array and store it in qrCode_array
        for x in range(width):
            qrCode_array[y][x] = pixel_array[y + min_y][x + min_x]

    qrCode = asarray(qrCode_array)
    d = decode(qrCode)
    data = d[0].data.decode()

    print()
    print("*" * 10)
    print("*" * 10)
    print("QR Code Decoded Successfully \n" + data)
    print("*" * 10)
    print("*" * 10)
    
    
    pyplot.show()

if __name__ == "__main__":
    main()
