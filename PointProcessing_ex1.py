import numpy as np
import cv2 
import matplotlib.pyplot as plt

def histogram(image):
    # Compute histogram of image
    # Return: array of length 256, where array[i] indicates the value of H(i)

    # If grayscale
    if len(image.shape) == 2:
        hist = np.empty(256, dtype=np.int)
        for i in range(256):
            hist[i] = np.sum(image == i)
        return hist

    # Compute and return hr, hg, hb for color image
    hist = np.empty((3, 256), dtype=np.int)
    for i in range(3):
        for j in range(256):
            hist[i,j] = np.sum(image[:,:,i] == j)
    return hist

def luminance(image):
    # If image is a grayscale image, return image
    if len(image.shape) == 2:
        return image
    # Convert color image to luminance image
    image = image.astype('float')
    L = 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]
    L = L.reshape((image.shape[0], image.shape[1])).astype('uint8')
    return L

def read_image(path, mode=0):
    # Read an image to a numpy array
    # Mode: 0 - Gray 1 - Color
    if mode == 0:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return image

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def main():
    # 1.1
    G = read_image("Lena.jpg", 0)
    C = read_image("Lena.jpg", 1)

    fig, ax = plt.subplots(2)
    fig.suptitle('Image')
    ax[0].imshow(G, cmap='gray')
    ax[0].set_title("Grayscale image")
    ax[1].imshow(C)
    ax[1].set_title("RGB image")
    plt.pause(2)

    # 1.2
    hist_G = histogram(G)
    hist_C = histogram(C)

    fig, ax = plt.subplots(2, 2)
    fig.suptitle('Histogram')
    ax[0, 0].bar(range(256), hist_G)
    ax[0, 0].set_title("hG")
    ax[0, 1].bar(range(256), hist_C[0])
    ax[0, 1].set_title("hr")
    ax[1, 0].bar(range(256), hist_C[1])
    ax[1, 0].set_title("hgr")
    ax[1, 1].bar(range(256), hist_C[2])
    ax[1, 1].set_title("hb")
    plt.pause(2)

    # plt.show()
    # 1.3
    L = luminance(C)
    plt.figure("Luminance image")
    plt.imshow(L, cmap='gray')
    plt.pause(2)
    # 1.4
    hist_L = histogram(L)

    hist_C = hist_C.astype('float')
    h = 0.299*hist_C[0] + 0.587*hist_C[1] + 0.114*hist_C[2]
    h = h.astype('uint8')

    fig, ax = plt.subplots(2)
    fig.suptitle('Image')
    ax[0].bar(range(256), hist_L)
    ax[0].set_title("hist_L")
    ax[1].bar(range(256), h)
    ax[1].set_title("h")
    plt.pause(2)

if __name__ == '__main__':
    main()