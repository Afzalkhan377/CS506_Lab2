import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

# Function to load and preprocess the image
def load_image(image_path):
        # Open the image using Pillow
    image = Image.open(image_path)
    # Convert the image to RGB mode if not already
    image = image.convert('RGB')
    # Convert the image to a NumPy array
    image_np = np.array(image)
    return image_np
 
# Function to perform KMeans clustering for image quantization
def image_compression(image_np, n_colors):
      # Get the dimensions of the image
    w, h, d = image_np.shape
    # Reshape the image to be a list of pixels (w*h, d)
    image_reshaped = image_np.reshape((w * h, d))
    
    # Apply KMeans clustering to reduce to n_colors
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(image_reshaped)
    
    # Get the colors of the centroids (cluster centers)
    compressed_colors = kmeans.cluster_centers_.astype('uint8')
    
    # Map each pixel to its corresponding cluster
    labels = kmeans.labels_
    
    # Create the compressed image by replacing each pixel with its corresponding cluster's color
    compressed_image = compressed_colors[labels].reshape((w, h, d))
    
    return compressed_image

# Function to concatenate and save the original and quantized images side by side
def save_result(original_image_np, quantized_image_np, output_path):
    # Convert NumPy arrays back to PIL images
    original_image = Image.fromarray(original_image_np)
    quantized_image = Image.fromarray(quantized_image_np)
    
    # Get dimensions
    width, height = original_image.size
    
    # Create a new image that will hold both the original and quantized images side by side
    combined_image = Image.new('RGB', (width * 2, height))
    
    # Paste original and quantized images side by side
    combined_image.paste(original_image, (0, 0))
    combined_image.paste(quantized_image, (width, 0))
    
    # Save the combined image
    combined_image.save(output_path)

def __main__():
    # Load and process the image
    image_path = 'favorite_image.png'  
    output_path = 'compressed_image.png'  
    image_np = load_image(image_path)

    # Perform image quantization using KMeans
    n_colors = 8  # Number of colors to reduce the image to, you may change this to experiment
    quantized_image_np = image_compression(image_np, n_colors)

    # Save the original and quantized images side by side
    save_result(image_np, quantized_image_np, output_path)
