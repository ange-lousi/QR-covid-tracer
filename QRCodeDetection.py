
from matplotlib import pyplot
from matplotlib.patches import Rectangle

import imageIO.png
import math
import statistics
import numpy


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
    mean_array = createInitializedGreyscalePixelArray(image_width, image_height)
    mean_filter = [ [1,  1,  1], 
                    [1,  1,  1], 
                    [1,  1,  1]]
    
    for x in range(1, image_height -1):
        for y in range(1, image_width -1):
            mean_array[x][y] =    ((mean_filter[0][0] * pixel_array[x-1][y-1]) + (mean_filter[0][1] * pixel_array[x-1][y]) + (mean_filter[0][2] * pixel_array[x-1][y+1]) +
                                   (mean_filter[1][0] * pixel_array[x][y-1])   +         pixel_array[x][y]                 + (mean_filter[1][2] * pixel_array[x][y+1]) + 
                                   (mean_filter[2][0] * pixel_array[x+1][y-1]) + (mean_filter[2][1] * pixel_array[x+1][y]) + (mean_filter[2][2] * pixel_array[x+1][y+1])) /9

    return mean_array

def convertBinary(pixel_array, threshHold, image_width, image_height):
    binaryArray = createInitializedGreyscalePixelArray(image_width, image_height);
    for i in range(image_height):
        for j in range(image_width):
            if  pixel_array[i][j] < threshHold:
                binaryArray[i][j] = 0
            else:
                binaryArray[i][j] = 1
    return binaryArray

def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    ConnectedComponentLabel = createInitializedGreyscalePixelArray(image_width, image_height)
    visited = createInitializedGreyscalePixelArray(image_width, image_height)
    currentLabel = 1
    d = {}
    for i in range(image_height):
        for j in range(image_width):

            if (pixel_array[i][j] > 0 and visited[i][j] == 0):
                q = Queue()
                q.enqueue([i, j])
                count = 0
                while not q.isEmpty():
                    n = q.dequeue()
                    ConnectedComponentLabel[n[0]][n[1]] = currentLabel
                    
                    if ((n[1]-1 >= 0) and (pixel_array[n[0]][n[1]-1] > 0) and (visited[n[0]][n[1]-1] == 0)):
                        q.enqueue([n[0], n[1]-1])
                        ConnectedComponentLabel[n[0]][n[1]-1] = currentLabel
                        visited[n[0]][n[1]-1] = 1
                        count += 1
                        
                    if ((n[1]+1 < image_width) and (pixel_array[n[0]][n[1]+1] > 0) and (visited[n[0]][n[1]+1] == 0) ):
                        q.enqueue([n[0], n[1]+1])
                        ConnectedComponentLabel[n[0]][n[1]+1] = currentLabel
                        visited[n[0]][n[1]+1] = 1
                        count += 1
                        
                    if (( n[0]-1 >= 0) and (pixel_array[n[0]-1][n[1]] > 0) and (visited[n[0]-1][n[1]] == 0)):
                        q.enqueue([n[0]-1, n[1]])
                        ConnectedComponentLabel[n[0]-1][n[1]] = currentLabel
                        visited[n[0]-1][n[1]] = 1
                        count += 1
                    if (( n[0]+1 < image_height) and (pixel_array[n[0]+1][n[1]] > 0) and (visited[n[0]+1][n[1]] == 0)):
                        q.enqueue([n[0]+1, n[1]])
                        ConnectedComponentLabel[n[0]+1][n[1]] = currentLabel
                        visited[n[0]+1][n[1]] = 1
                        count += 1
                        
                d.update({currentLabel: count})
                currentLabel += 1

    v = list(d.values())
    k = list(d.keys())

    mostLabels = k[v.index(max(v))]
    mostLabelledImage = createInitializedGreyscalePixelArray(image_width, image_height)

    for x in range(image_height):
        for y in range(image_width):
            
            if (ConnectedComponentLabel[x][y] == mostLabels):
                mostLabelledImage[x][y] = ConnectedComponentLabel[x][y]
            else:
                mostLabelledImage[x][y] = 0
                
    return mostLabelledImage    

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
    

def main():
    filename = "./images/covid19QRCode/poster1small.png"

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(filename)
    
    #STEP 1
    greyRGBimage = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    scaledGreyRGBimage = contrastStretch(greyRGBimage, image_width, image_height)
    #pyplot.imshow(scaledGreyRGBimage, cmap="gray")

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
    position = []
    for m in range(image_height):
        for n in range(image_width):
            if largestConnectedComponent[m][n] > 0:
                position.append([m , n])          
    minX = position[0][0]
    minY = position[0][1]
    Xmax = position[-1][0]
    Ymax = position[-1][1]
    
    width = Xmax - minX
    height = Ymax - minY
    
    axes = pyplot.gca()
    rect = Rectangle( (minX-30, minY+35), width, height, linewidth=3, edgecolor='g', facecolor='none' )
    axes.add_patch(rect)
    
    pyplot.imshow(prepareRGBImageForImshowFromIndividualArrays(px_array_r, px_array_g, px_array_b, image_width, image_height))
    pyplot.show()


if __name__ == "__main__":
    main()
