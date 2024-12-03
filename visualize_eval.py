import os
from PIL import Image
import argparse

'''
Compose the rendered images into large grids
'''

def compose_images(img_dir, num_large, rows, cols, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    img_files = [os.path.join(img_dir, img) for img in os.listdir(img_dir)
                if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    total_images = num_large * rows * cols
    img_files = img_files[:total_images]
    # print(img_files)
    for i in range(num_large):
        grid_images = img_files[i*rows*cols : (i+1)*rows*cols]
        images = [Image.open(img) for img in grid_images]
        widths, heights = zip(*(img.size for img in images))
        single_width = max(widths)
        single_height = max(heights)
        large_image = Image.new('RGB', (cols * single_width, rows * single_height))

        for idx, img in enumerate(images):
            x = (idx % cols) * single_width
            y = (idx // cols) * single_height
            large_image.paste(img, (x, y))

        large_image.save(os.path.join(output_dir, f'large_image_{i+1}.jpg'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compose images into large grids.')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory of images.')
    parser.add_argument('--num_large', type=int, required=True, help='Number of large images to create.')
    parser.add_argument('--rows', type=int, required=True, help='Number of rows in each large image.')
    parser.add_argument('--cols', type=int, required=True, help='Number of columns in each large image.')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save large images.')

    args = parser.parse_args()
    compose_images(args.img_dir, args.num_large, args.rows, args.cols, args.img_dir)