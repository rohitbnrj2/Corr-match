"""
Simulate conversion of RGB to photon-image
"""
from loguru import logger

try:

    import imageio.v2 as io
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt

except ImportError as e:
    logger.error(
        "Install Imageio, OpenCV, Matplotlib, and NumPy"
    )
    raise e


def luminance_img(img: np.ndarray) -> np.ndarray:
    """
    Convert an sRGB image to RGB & then to luminance.

    Args:
        img (np.ndarray): Input sRGB image.

    Returns:
        np.ndarray: Luminance image.
    """

    # Convert sRGB to RGB
    img = img.astype(np.float64) / 255.0
    mask = img <= 0.04045
    img = np.where(
        mask, img / 12.92, ((img + 0.055) / 1.055) ** 2.4
    )

    # Convert linear RGB to luminance
    lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    lum = np.clip(lum, 0, 1).astype(np.float64)

    return lum


def convert_to_8bit(img: np.ndarray) -> np.ndarray:
    """
    Convert luminance image to 8-bit quantized image.

    Args:
        img (np.ndarray): Input luminance image.

    Returns:
        np.ndarray: 8-bit quantized image.    
    """

    try:
        img *= 255.0
        quant_img = np.clip(img, 0, 255).astype(np.uint8)

    except Exception as e:
        logger.error("Error converting image to 8-bit: {}", e)
        raise e

    return quant_img


def convert_to_photon(img: np.ndarray,
                      max_photons: float = 0.6,
                      pde: float = 0.6,
                      dark_count_rate: float = 2e2,
                      exposure_time: float = 1e-8,
                      ) -> np.ndarray:
    """
    Convert 8-bit image to photon image.

    Args:
        img (np.ndarray): Input 8-bit image.
        max_photons (int): Maximum number of photons.
        pde (float): Photon detection efficiency.
        dark_count_rate (float): Dark count rate.
        exposure_time (float): Exposure time.

    Returns:
        np.ndarray: Photon image.
    """

    # Set random seed
    np.random.seed(42)

    # Check img is 8-bit
    if img.dtype != np.uint8:
        raise ValueError("Input image must be of type uint8")

    # Normalize intensities to get luminance
    norm_img = (img / 255.0).astype(np.float32)

    # Dark counts per exposure
    lambda_dark = dark_count_rate * exposure_time

    # Expected photons per pixel
    lambda_signal = norm_img * max_photons * pde

    # Total expected photons - non-negative
    total_lambda = np.clip(lambda_signal + lambda_dark, 0, None)

    # Detection as Poisson
    prob = 1 - np.exp(-total_lambda)
    photon_img = (np.random.rand(*img.shape) < prob).astype(np.uint8)
    return photon_img


def visualize(**kwargs) -> None:
    """
    Method to visualize a set of images.

    Args:
        **kwargs: Keyword arguments containing image names and their corresponding arrays.
    """

    # Number of images 
    n = len(kwargs)
    if n == 0:
        logger.warning("No images provided for visualization.")
        return

    # Setup figure and plots
    fig, axs = plt.subplots(1, n, figsize=(15, 10))
    if n == 1:
        axs = [axs]

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # Plot each image
    for i, (name, img) in enumerate(kwargs.items()):

        if isinstance(img, np.ndarray) is False:
            logger.warning(
                f"Image for {name} is not numpy array"
            )
            raise ValueError(f"Image for {name} is not numpy array")

        if img.ndim == 2:
            axs[i].imshow(img, cmap='gray')
        else:
            axs[i].imshow(img)

        axs[i].axis('off')
        axs[i].set_title(f"{name.capitalize()}", fontsize=10)

    # Save the figure with the combined images
    plt.tight_layout()
    plt.savefig("Output.png", bbox_inches='tight')
    plt.close()


def main():
    """
    Simulate conversion of RGB to photon-image
    """

    # Load an example image
    img = io.imread("plant.jpg")

    # Convert image to Luminance
    lum_img = luminance_img(img)

    # Quantize to 8-bit
    quantize_img = convert_to_8bit(lum_img)

    # Convert to Photon image
    photon_img = convert_to_photon(quantize_img)

    # Visualize Images
    visualize(
        original=img, luminance=lum_img,
        quantized=quantize_img, photon=photon_img
    )

    logger.info("Photon image simulation completed.")

if __name__ == "__main__":
    main()

