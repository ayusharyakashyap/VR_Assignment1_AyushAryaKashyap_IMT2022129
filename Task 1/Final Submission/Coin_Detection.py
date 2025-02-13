import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("IndianCoins.jpg")
original = img.copy()  # Keeping a copy for displaying results

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blur = cv2.GaussianBlur(gray, (7, 7), 0)

# Apply Canny edge detection
canny = cv2.Canny(blur, 90, 255)

# Find contours
contours, _ = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Count total number of coins
total_coins = len(contours)

# Draw contours on the original image
cv2.drawContours(original, contours, -1, (0, 255, 0), 2)

# Save the processed images
cv2.imwrite("Detected_Coins.jpg", original)  # Image with drawn contours
cv2.imwrite("Grayscale.jpg", gray)          # Grayscale image
cv2.imwrite("Blurred.jpg", blur)            # Blurred image
cv2.imwrite("Canny_Edges.jpg", canny)       # Edge-detected image

# Convert images to RGB for matplotlib display
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
blur_rgb = cv2.cvtColor(blur, cv2.COLOR_GRAY2RGB)
canny_rgb = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)

# Display images using matplotlib
plt.figure(figsize=(15, 10))

# Original Image with Contours and Coin Count
plt.subplot(1, 3, 1)
plt.imshow(original_rgb)
plt.title(f"Detected Coins (Total: {total_coins})")
plt.axis('off')

# Grayscale Image
plt.subplot(1, 3, 2)
plt.imshow(blur_rgb)
plt.title("Blurred Image")
plt.axis('off')

# Canny Edge Detection Image
plt.subplot(1, 3, 3)
plt.imshow(canny_rgb, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis('off')

plt.tight_layout()
plt.show()

# Print total coin count
print(f"Total number of coins detected: {total_coins}")
print("Processed images saved successfully.")
