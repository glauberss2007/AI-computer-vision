# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Open an image file using the Pillow library (PIL)
pic = Image.open('./files/00-puppy.jpg')

# Check the type of the 'pic' variable (PIL Image object)
type(pic)

# Convert the PIL Image to a NumPy array
pic_arr = np.asarray(pic)

# Check the type of the 'pic_arr' variable (NumPy array)
type(pic_arr)

# Print the shape of the NumPy array representing the image
print(pic_arr.shape)

# Display the original image using Matplotlib
plt.imshow(pic_arr)
plt.show()

# Create a copy of the NumPy array representing the image
pic_red = pic_arr.copy()

# Print the shape of the copied array
print(pic_red.shape)

# Display the green channel of the copied image
plt.imshow(pic_red[:,:,1])
plt.show()

# Display the red channel of the copied image in grayscale
# Note: cmap='gray' specifies the colormap to be grayscale
plt.imshow(pic_red[:,:,0], cmap='gray')
plt.show()
