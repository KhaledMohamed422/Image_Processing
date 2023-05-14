from PIL import Image

def direct_mapping_resize(image_path, new_width, new_height):
    # Open the image
    image = Image.open(image_path)

    # Create a new image with the desired dimensions
    new_image = Image.new(mode="L", size=(new_width, new_height))

    # Calculate the scaling factors
    width_ratio = new_width / image.width
    height_ratio = new_height / image.height

    # Map each pixel from the source image to the resized image
    for y in range(new_height):
        for x in range(new_width):
            # Find the corresponding pixel in the source image
            source_x = int(x / width_ratio)
            source_y = int(y / height_ratio)

            # Get the grayscale intensity of the corresponding pixel
            pixel = image.getpixel((source_x, source_y))
            intensity = pixel if isinstance(pixel, int) else pixel[0]

            # Set the intensity of the current pixel in the new image
            new_image.putpixel((x, y), intensity)

    return new_image

# Example usage
input_image = "D:\FCI\الفرقة الثالثة\الترم الثاني\جديد\Digital Image Processing\Tasks\ChatGPT\input_image.jpg"
output_image_path = "D:\FCI\الفرقة الثالثة\الترم الثاني\جديد\Digital Image Processing\Tasks\ChatGPT\output_image.jpg"
resized_image = direct_mapping_resize(input_image, 500, 500)
resized_image.save(output_image_path)
