import argparse
import os
from PIL import Image


def resize_image(image, size):
	return image.resize(size, Image.ANTIALIAS)

def resize_images(image_dir, output_dir, size):
	"""resize the images in image_dir and save to output_dir"""
	images = os.listdir(image_dir)  #returns a list containing name of the entries(i.e.,name of images) in the given path(directory)
	num_images = len(images)
	for i, image in enumerate(images):
		with open(os.path.join(image_dir, image), 'r+b') as f:
			with Image.open(f) as img:
				img = resize_image(img, size)
				img.save(os.path.join(output_dir, image), img.format)

		if (i+1) % 100 ==0:
			print("[{}/{}] resized the images and saved to '{}'.".format(i+1,num_images, output_dir))
	
def main(args):
	image_dir = args.image_dir
	output_dir = args.output_dir
	image_size = [args.image_size, args.image_size]
	resize_images(image_dir, output_dir, image_size)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--image_dir', type=str, default='/home/welcome/pytorch-tutorial/tutorials/03-advanced/image_captioning/data/train2014/', help="directory for train images")
	parser.add_argument('--output_dir', type=str, default='/home/welcome/pytorch-tutorial/tutorials/03-advanced/image_captioning/data/resized2014/', help="directory for saving resized images")
	parser.add_argument('--image_size', type=int, default=224, help="size for image after processing")
	args = parser.parse_args()
	main(args)
