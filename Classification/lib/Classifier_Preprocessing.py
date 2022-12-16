from pathlib import Path
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt

import os

__file__ = 'Preprocessing.ipynb'
BASE_DIR = Path(__file__).resolve().parent


def slice_number(numbers, axis, np_array_img):
    """
    Select the number of layers according to the number table and the axis from the center of the MRI

    Args:
        numbers ([int]): number of slices to be extracted in positive
        axis (int): x = 0, y = 1, z = 2
        np_array_img : np array du mri

    Returns:
        ([]) : table of slices selected numbers
    """

    # Calculation of the length according to the axis of the MRI
    if axis == 2:
        length = len(np_array_img[1][1])
    else:
        length = len(np_array_img[1])

    slice_numbers = []
    for i, number in enumerate(numbers):
        if i == 1:
            slice_numbers.append(int(length / 2))
        slice_numbers.append(int(length / 2) + number)
        slice_numbers.append(int(length / 2) - number)
    return slice_numbers


def create_slices(slice_numbers: [int], axis: int, np_array_img) -> []:
    """
    Extract the slices from the MRI according to the axis

    Args:
        slice_numbers ([int]): slice numbers
        axis (int): x = 0, y = 1, z = 2
        np_array_img : np array du mri

    Returns:
        ([]) : tab of the slices in np_array_img
    """
    slices = []
    for i, number in enumerate(slice_numbers):
        if axis == 0:
            slices.append(np_array_img[number, :, :])
        elif axis == 1:
            slices.append(np_array_img[:, number, :])
        else:
            slices.append(np_array_img[:, :, number])
    return slices


def show_slices(slices_x: [], slices_y: [], slices_z: []):
    """
    Show slices selected into a subplot

    Args:
        slices_x ([]): tab slices of x-axis
        slices_y ([]): tab slices of y-axis
        slices_z ([]): tab slices of z-axis
    """

    columns = len(slices_x)

    fig, axes = plt.subplots(3, columns)
    for i, slice in enumerate(slices_x):
        axes[0, i].imshow(slice.T, cmap="gray", origin="lower")
    for i, slice in enumerate(slices_y):
        axes[1, i].imshow(slice.T, cmap="gray", origin="lower")
    for i, slice in enumerate(slices_z):
        axes[2, i].imshow(slice.T, cmap="gray", origin="lower")
    plt.show()


def save_slices(slices: [], axis: int):
    """
    Save the slices in directory save/axis_*/ according to the axis

    Args:
        slices ([]): slices array of images to save
        axis (int): x = 0, y = 1, z = 2

    Returns:
        (bool) : Return True if the images are saved and False if there are an error
    """
    base_path = "save/"
    if axis == 0:
        base_path = base_path + "x_axis/"
        _axis = "x"
    elif axis == 1:
        base_path = base_path + "y_axis/"
        _axis = "y"
    else:
        base_path = base_path + "z_axis/"
        _axis = "z"

    try:
        for i, slice in enumerate(slices):
            path = base_path + f"slice_{_axis}_{i}.jpeg"
            plt.imsave(path, np.rot90(slice), cmap="gray")
        return True
    except:
        return False


def save_all_slices(slices_x: [], slices_y: [], slices_z: [], path_dir='save/'):
    """
    Save all the slices in directory save/axis_*/

    Args:
        slices_x ([]): tab slices of x-axis
        slices_y ([]): tab slices of y-axis
        slices_z ([]): tab slices of z-axis

    Returns:
        (bool) : Return True if the images are saved and False if there are an error
    """
    path_x = f"{path_dir}/x_axis"
    path_y = f"{path_dir}/y_axis"
    path_z = f"{path_dir}/z_axis"

    if not os.path.exists(path_x):
        os.makedirs(path_x)
    if not os.path.exists(path_y):
        os.makedirs(path_y)
    if not os.path.exists(path_z):
        os.makedirs(path_z)

    try:
        for i, slice in enumerate(slices_x):
            path = path_x + f"/slice_x_{i}.jpeg"
            plt.imsave(path, np.rot90(slice), cmap="gray")
        for i, slice in enumerate(slices_y):
            path = path_y + f"/slice_y_{i}.jpeg"
            plt.imsave(path, np.rot90(slice), cmap="gray")
        for i, slice in enumerate(slices_z):
            path = path_z + f"/slice_z_{i}.jpeg"
            plt.imsave(path, np.rot90(slice), cmap="gray")
        return True
    except:
        return False


def crop_slices(current_slice: []) -> []:
    """
    Crop 1 slice in np array

    Args:
        current_slice ([]): tab of 1 slice in np array

    Returns:
        ([]) : Return the np array cropped
    """
    length = len(current_slice[:])
    # x_start
    for x in range(0,length):
        if sum(current_slice[x]) > 0:
            x_start = x
            break
    # x_end
    for x in range(length-1,0,-1):
        if sum(current_slice[x]) > 0:
            x_end = x
            break

    length = len(current_slice[0,:])
    # y_start
    for y in range(0,length):
        if sum(current_slice[:, y]) > 0:
            y_start = y
            break
    # y_end
    for y in range(length-1,0,-1):
        if sum(current_slice[:, y]) > 0:
            y_end = y
            break
    return current_slice[x_start:x_end, y_start:y_end]


if __name__ == "__main__":
    # Find path of the different MRI
    path_flair = BASE_DIR / "BraTS20_Training_001/BraTS20_Training_001_flair.nii"
    path_t1 = BASE_DIR / "BraTS20_Training_001/BraTS20_Training_001_t1.nii"
    path_t1ce = BASE_DIR / "BraTS20_Training_001/BraTS20_Training_001_t1ce.nii"
    path_t2 = BASE_DIR / "BraTS20_Training_001/BraTS20_Training_001_t2.nii"

    # Transform the MRI t1ce into a numpy array : np_array_img
    nii_img = nib.load(path_t1ce)
    nii_img_data = nii_img.get_fdata()
    np_array_img = np.array(nii_img_data)

    # Select the number of layers according to the number table and the axis from the center of the MRI
    slice_numbers_x = slice_number([5, 10], axis=0, np_array_img=np_array_img)
    slice_numbers_y = slice_number([5, 10], axis=1, np_array_img=np_array_img)
    slice_numbers_z = slice_number([5, 10], axis=2, np_array_img=np_array_img)

    # Extract the slices from the MRI according to the axis
    slices_x = create_slices(slice_numbers_x, axis=0, np_array_img=np_array_img)
    slices_y = create_slices(slice_numbers_y, axis=1, np_array_img=np_array_img)
    slices_z = create_slices(slice_numbers_z, axis=2, np_array_img=np_array_img)

    # Show the subplot of the slices
    show_slices(slices_x[:], slices_y[:], slices_z[:])

    # Crop the slices
    for nb_slices in range(0, len(slices_x)):
        slices_x[nb_slices] = crop_slices(slices_x[nb_slices])
    for nb_slices in range(0, len(slices_y)):
        slices_y[nb_slices] = crop_slices(slices_y[nb_slices])
    for nb_slices in range(0, len(slices_z)):
        slices_z[nb_slices] = crop_slices(slices_z[nb_slices])

    # Show the subplot of the slices
    show_slices(slices_x[:], slices_y[:], slices_z[:])

    # Save the images on the file save/
    print(save_all_slices(slices_x[:], slices_y[:], slices_z[:]))
    # save_slices(slices[:], axis=0)
    
    
    # PREDICTION
    classifier = tf.keras.models.load_model('my_model.h5')
    
    SAVE_DIR = Path(BASE_DIR / 'save/')
    save_dir = os.listdir(str(SAVE_DIR))
    
    tab = []
    for dir in save_dir:
        path = SAVE_DIR / dir
        for f in Path(path).glob("*.jepg") :
            tab.append(f)
    
    test_image = []
    for i in range():
        test_image.append(keras.preprocessing.image.load_img(tab[i], target_size = (128,128)))
    
    resultTab = []
    for i in range(len(test_image)) :
        image = keras.utils.img_to_array(test_image[i])
        image = np.expand_dims(image, axis=0)
        result = model.predict(image)
        resultTab.append(result)
        
    result = []
    for i in range(len(resultTab)):
        max = 0
        for j in range(4):
            if max < resultTab[i][0][j]:
                max = j
        if max == 0:
            result.append('glioma')
        elif max == 1:
            result.append('meningioma')
        elif max == 2:
            result.append('notumor')
        elif max == 3:
            result.append('pituitary')
    
    #counter.most_common(nombre_classement)
