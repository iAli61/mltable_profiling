# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script contains methods to hangle inputs for tensorflow model training.
"""

import glob
import logging
import re

import tensorflow
import mltable
import os

class ImageAndMaskHelper:
    """Helps locating images and masks for training a segmentation model"""

    def __init__(self, image_ds, mask_ds, images_type: str = "png"):
        """Initialize the helper class.

        Args:
            mltable_file_path (str): Path to the MLTable file.
        """
        self.logger = logging.getLogger(__name__)
        self.images_type = images_type
        self.image_masks_pairs = []
        self.mask_ds = mask_ds
        self.image_ds = image_ds

        # load mltable
        self.logger.info(f"[IMAGE] Loading MLTable from {image_ds}")
        self.logger.info(f"[MASK] Loading MLTable from {mask_ds}")
        
        

    def build_pair_list(self):
        """Builds a list of pairs of paths to image/mask from the dataframe.

        Returns:
            image_masks_pairs (List[tuple(str, str)])
        """

        parsing_stats = {
            "masks_not_matching": 0,
            "images_not_matching": 0,
            "images_without_masks": 0,
        }
        # search for all masks matching file name pattern
        masks_filename_pattern = re.compile("(.*)\\.png")

        masks_paths = []
        for file_path in glob.glob(self.mask_ds + "/**/*", recursive=True):
            matches = masks_filename_pattern.match(os.path.basename(file_path))
            if matches:
                masks_paths.append((matches.group(1), file_path))
            else:
                # keep some stats
                parsing_stats["masks_not_matching"] += 1
        masks_paths = dict(masks_paths)  # turn list of tuples into a map

        # search for all images matching file name pattern
        images_filename_pattern = re.compile("(.*)\\.jpg")
        images_paths = []
        for file_path in glob.glob(self.image_ds + "/**/*", recursive=True):
            matches = images_filename_pattern.match(os.path.basename(file_path))
            if matches:
                images_paths.append((matches.group(1), file_path))
            else:
                # keep some stats
                parsing_stats["images_not_matching"] += 1

        # now match images and masks
        self.images = []  # list of images
        self.masks = []  # list of masks (ordered like self.images)
        self.image_masks_pairs = []  # list of tuples

        for image_key, image_path in images_paths:
            if image_key in masks_paths:
                self.images.append(image_path)
                self.masks.append(masks_paths[image_key])
                self.image_masks_pairs.append((image_path, masks_paths[image_key]))
            else:
                self.logger.debug(
                    f"Image {image_path} doesn't have a corresponding mask."
                )
                # keep some stats
                parsing_stats["images_without_masks"] += 1

        parsing_stats["found_pairs"] = len(self.image_masks_pairs)

        self.logger.info(f"Finished parsing images/masks paths: {parsing_stats}")

        return self.image_masks_pairs

    def __len__(self):
        return len(self.image_masks_pairs)

class ImageAndMaskSequenceDataset(ImageAndMaskHelper):
    """Creates a tensorflow.data.Dataset out of a list of images/masks"""

    @staticmethod
    def loading_function(image_path, mask_path, target_size, image_type="png"):
        """Function called using tf map() for loading images into tensors"""
        # logging.getLogger(__name__).info(f"Actually loading image {image_path}")
        image_content = tensorflow.io.read_file(image_path)
        if image_type == "jpg":
            image = tensorflow.image.decode_jpeg(image_content, channels=3)
        else:
            image = tensorflow.image.decode_png(image_content, channels=3)
        image = tensorflow.image.resize(image, [target_size, target_size])
        image = tensorflow.image.convert_image_dtype(image, tensorflow.float32)

        mask_content = tensorflow.io.read_file(mask_path)
        mask = tensorflow.image.decode_png(
            mask_content, channels=1, dtype=tensorflow.dtypes.uint8
        )
        mask = tensorflow.image.resize(
            mask, [target_size, target_size], antialias=False
        )
        mask = tensorflow.math.round(mask)
        mask -= 1
        # mask = tensorflow.image.convert_image_dtype(mask, tensorflow.uint8)

        return image, mask

    def dataset(self, input_size=160):
        """Creates a tf dataset and a loading function."""
        # builds the list of pairs
        self.build_pair_list()

        # https://cs230.stanford.edu/blog/datapipeline/#best-practices
        with tensorflow.device("CPU"):
            _dataset = tensorflow.data.Dataset.from_tensor_slices(
                (self.images, self.masks)
            )
            _loading_function = (
                lambda i, m: ImageAndMaskSequenceDataset.loading_function(
                    i, m, input_size, image_type=self.images_type
                )
            )

        return _dataset, _loading_function
