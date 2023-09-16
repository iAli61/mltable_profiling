# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script contains methods to hangle inputs for tensorflow model training.
"""

import logging

import tensorflow
import mltable

class ImageAndMaskHelper:
    """Helps locating images and masks for training a segmentation model"""

    def __init__(self, dataset, images_type: str = "png"):
        """Initialize the helper class.

        Args:
            mltable_file_path (str): Path to the MLTable file.
        """
        self.logger = logging.getLogger(__name__)
        self.images_type = images_type
        self.image_masks_pairs = []
        self.images = []
        self.masks = []

        # load mltable
        tbl = mltable.load(dataset)
        # load into pandas
        self.tdf = tbl.to_pandas_dataframe()
        

    def build_pair_list(self):
        """Builds a list of pairs of paths to image/mask from the dataframe.

        Returns:
            image_masks_pairs (List[tuple(str, str)])
        """
        for index, row in self.tdf.iterrows():
            image_path = row['image_url']
            mask_path = row['mask_url']
            
            self.images.append(image_path)
            self.masks.append(mask_path)
            self.image_masks_pairs.append((image_path, mask_path))

        self.logger.info(f"Finished parsing images/masks paths. Found {len(self.image_masks_pairs)} pairs.")
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
