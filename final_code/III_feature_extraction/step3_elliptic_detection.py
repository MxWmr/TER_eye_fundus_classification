import matplotlib.pyplot as plt
from skimage import data, color, io, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
import cv2

# Load picture, convert to grayscale and detect edges
# image_rgb = data.coffee()[0:220, 160:420] image de test
img = io.imread('reduced_dataset/Train/Glaucoma_Negative/001.jpg')
scale_percent = 40 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
image_rgb = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
image_gray = color.rgb2gray(image_rgb)


edges = canny(image_gray, sigma=2,
              low_threshold=0.55, high_threshold=0.8)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))



ax.imshow(edges, cmap='gray')
ax.set_title(r'Canny filter, $\sigma=2$', fontsize=20)

ax.axis('off')

fig.tight_layout()
plt.show()


# Perform a Hough Transform
# The accuracy corresponds to the bin size of a major axis.
# The value is chosen in order to get a single high accumulator.
# The threshold eliminates low accumulators
result = hough_ellipse(image_gray, accuracy=5, threshold=150,
                       min_size=10, max_size=280)
result.sort(order='accumulator')

try:
    # Estimated parameters for the ellipse
    best = list(result[-1])
    yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    orientation = best[5]

    # Draw the ellipse on the original image
    cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
    image_rgb[cy, cx] = (0, 0, 255)
    # Draw the edge (white) and the resulting ellipse (red)
    edges = color.gray2rgb(img_as_ubyte(image_rgb))
    edges[cy, cx] = (250, 0, 0)

    fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), sharex=True,
                                    sharey=True,
                                    subplot_kw={'adjustable':'box'})

    ax1.set_title('Original picture')
    ax1.imshow(image_rgb)

    ax2.set_title('Edge (white) and result (red)')
    ax2.imshow(edges)

    plt.show()

except:
    print('No ellipse detected')
