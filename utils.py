import numpy as np
from skimage import io, img_as_float
from PIL import Image


def get_dark_channel(image, window_size):
    gray = np.min(image, axis=2)
    dark_channel = np.zeros_like(gray)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            top = max(0, i - window_size // 2)
            left = max(0, j - window_size // 2)
            #bottom = min(image.shape[0], i + window_size // 2)
            #right = min(image.shape[1], j + window_size // 2)
            bottom = max(top + 1, min(image.shape[0], i + window_size // 2))
            right = max(left + 1, min(image.shape[1], j + window_size // 2))
            dark_channel[i, j] = np.min(gray[top:bottom, left:right])
    return dark_channel


def get_atmosphere(image, dark_channel):
    dark_channel_vec = dark_channel.ravel()
    indices = dark_channel_vec.argsort()[-int(len(dark_channel_vec) * 0.001):]
    atmosphere = np.zeros(3)
    for i in indices:
        atmosphere += image.reshape(-1, 3)[i]
    atmosphere /= len(indices)
    return atmosphere


def dehaze(image, atmosphere, t_min=0.1, window_size=15):
    transmission = 1 - 0.95 * get_dark_channel(image / atmosphere, window_size)
    transmission = np.maximum(transmission, t_min)
    transmission_norm = (transmission - np.min(transmission)) / (np.max(transmission) - np.min(transmission))
    dehazed = np.zeros_like(image)
    for i in range(3):
        dehazed[:, :, i] = (image[:, :, i] - atmosphere[i]) / transmission_norm + atmosphere[i]
    return np.clip(dehazed, 0, 1)


def process_dehazing(image_path):
    image = io.imread(image_path)
    image = img_as_float(image[:, :, :3])
    dark_channel = get_dark_channel(image, window_size=15)
    atmosphere = get_atmosphere(image, dark_channel)
    dehazed = dehaze(image, atmosphere)

    # Convert numpy array back to PIL Image and return
    dehazed_image = Image.fromarray((dehazed * 255).astype(np.uint8))
    return image, dehazed_image
