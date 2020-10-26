import numpy as np
import cv2 
import matplotlib.pyplot as plt

rgb_luminance_coeff = np.asarray([0.299, 0.587, 0.114])

def change_brightness(image, beta=0.):
    if len(image.shape) == 2:
        return np.clip(image + beta, 0, 255).astype('uint8')
    rgb_beta = rgb_luminance_coeff * beta
    return np.clip(image + rgb_beta, 0, 255).astype('uint8')
    
def change_contrast(image, alpha=1.):
    return np.clip(image.astype('float')*alpha, 0, 255).astype('uint8')

def change_gamma(image, gamma=1.):
    image = 255. * ((image.astype('float') / 255.) ** (1. / gamma))
    return np.clip(image, 0, 255).astype('uint8')
    
def contrast_stretch(image, min=0, max=255):
    image = ((image.astype('float') - min) / (max - min)) * 255.
    return image.astype('uint8') 

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
    image = read_image("Lena.jpg", 1)
    image_brightness = change_brightness(image, -40)
    image_contrast = change_contrast(image, 1.5)
    image_gamma = change_gamma(image, 0.5)
    image_stretch = contrast_stretch(image, 10, 180)

    # Plot results
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(20,4))
    fig.suptitle('Bai 2')
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[1].imshow(image_brightness)
    ax[1].set_title("Brightness Image")
    ax[2].imshow(image_contrast)
    ax[2].set_title("Contrast Image")
    ax[3].imshow(image_gamma)
    ax[3].set_title("Gamma Image")
    ax[4].imshow(image_stretch)
    ax[4].set_title("Stretch Image")
    plt.show()

if __name__ == '__main__':
    main()