"""
Copyright (C) 2022-present  SauceyRed (42098474+SauceyRed@users.noreply.github.com)
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import cv2
import argparse

from os import listdir, path
from time import time
from math import floor

parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("comparison_dir")
parser.add_argument("--verbose", "-v", "-V", action="store_true")

comparisons = {}

def CalcDuration(duration):
	total_minutes = duration / 60
	seconds = floor(duration % 60)
	hours = floor(total_minutes / 60)
	minutes = floor(total_minutes % 60)
	if seconds < 10: seconds = "0" + str(seconds)
	if hours < 10: hours = "0" + str(hours)
	if minutes < 10: minutes = "0" + str(minutes)
	return hours, minutes, seconds

def CompareImages(test_image_path, comp_imgs_dir, verbose):
	test_image = cv2.imread(test_image_path)
	test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
	test_image_hist = cv2.calcHist([test_image_gray], [0], None, [256], [0, 256])

	start_time = time()

	image_count = 1

	for image in listdir(comp_imgs_dir):
		if not image.endswith((".png", ".jpg")):
			if verbose: print(f"File {image} is not an image, skipping...")
			continue
		if verbose: print(f"Image: {image} ({image_count}/{len(listdir(comp_imgs_dir))})")
		comp_image = cv2.imread(path.join(comp_imgs_dir, image))
		comp_image_gray = cv2.cvtColor(comp_image, cv2.COLOR_BGR2GRAY)
		comp_image_hist = cv2.calcHist([comp_image_gray], [0], None, [256], [0, 256])

		comp = 0

		i = 0
		while i < len(test_image_hist) and i < len(comp_image_hist):
			comp += (test_image_hist[i] - comp_image_hist[i])**2
			i += 1
		comp = comp**(1 / 2)

		comparisons[image] = comp

		image_count += 1

	duration = time() - start_time

	hours, minutes, seconds = CalcDuration(duration)

	if verbose: print(f"Process duration: {hours}:{minutes}:{seconds}")

	return min(comparisons, key=comparisons.get)

if __name__ == "__main__":
	args = parser.parse_args()
	print(CompareImages(args.image, args.comparison_dir, args.verbose))
