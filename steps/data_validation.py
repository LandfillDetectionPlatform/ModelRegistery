from PIL import Image
import numpy as np
from zenml import step

@step
def data_validation(data_path: str) -> bool:
    """Validates that the image at data_path is not entirely black or white.

    Args:
    data_path (str): The file path to the image to be validated.

    Returns:
    bool: True if the image passes validation, False otherwise.
    """
    # Load the image using PIL
    img = Image.open(data_path)

    # Convert the image to a NumPy array
    img_array = np.array(img)

    # If the image has more than one channel, consider it as a colored image
    # and convert it to grayscale to simplify the validation
    if img_array.ndim > 2:
        img_array = img_array.mean(axis=2)

    # Check if all pixels are the same (either all 0s for black or all 255s for white)
    unique_values = np.unique(img_array)
    if len(unique_values) == 1 and unique_values[0] in [0, 255]:
        # The image is entirely black or white
        return False

    # The image passes validation
    return True
