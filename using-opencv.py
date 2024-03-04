# Import necessary libraries
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
import cv2

# Read an image using OpenCV
img = cv2.imread('./files/00-puppy.jpg')

# Print the type of the 'img' variable (OpenCV uses BGR format)
print(type(img))

# Display the original image using Matplotlib
plt.imshow(img)
plt.show()

# Convert the image from BGR to RGB using OpenCV
fix_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image with correct color using Matplotlib
plt.imshow(fix_img)
plt.show()

# Read the image in grayscale using OpenCV
img_gray = cv2.imread('./files/00-puppy.jpg', cv2.IMREAD_GRAYSCALE)

# Display the grayscale image using Matplotlib
plt.imshow(img_gray, cmap='gray')
plt.show()

# Resize the image to a specified width and height using OpenCV
new_img_resized = cv2.resize(fix_img, (1000, 400))

# Display the resized image using Matplotlib
plt.imshow(new_img_resized)
plt.show()

# Resize the image based on a specified ratio using OpenCV
w_ratio = 0.5
h_ratio = 0.5
new_img_ratio = cv2.resize(fix_img, (0, 0), fix_img, w_ratio, h_ratio)

# Display the resized image based on ratio using Matplotlib
plt.imshow(new_img_ratio)
plt.show()

# Flip the image vertically using OpenCV
new_img_flipped = cv2.flip(fix_img, 0)

# Display the flipped image using Matplotlib
plt.imshow(new_img_flipped)
plt.show()

# Print the type of the 'fix_img' variable
type(fix_img)

# Save the fixed image as a new file using OpenCV
cv2.imwrite('tatlNew.jpg', fix_img)
