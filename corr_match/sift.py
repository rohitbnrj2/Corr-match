"""
SIFT (Scale-Invariant Feature Transform) implementation for feature matching 
to find correspondences between images. Using OpenCV.
"""
from __future__ import annotations

from typing import Dict, List

import os
import os.path as osp
import random
import cv2

import numpy as np
from loguru import logger
from omegaconf import DictConfig

# Current Directory
curr_dir = osp.abspath(osp.dirname(__file__))


def sift_runner(cfgs: DictConfig) -> None:
    """
    Run SIFT on the images provided in the dataset.
    """

    # Create SIFT runner object
    runner = Sift(cfgs)

    # Get correspondence matching for image pairs in the dataset
    runner()



class Sift:
    """
    Feature matching with SIFT.
    """

    def __init__(self, cfgs: DictConfig) -> None:

        self._cfg = cfgs
        self.d_name = self._cfg.exp.dataset_name
        self.indices = self.get_indices()


    def __call__(self, ) -> None:
        """
        Run SIFT to get correspondence between pairs of images.
        """

        # Run the loop
        for i, f_name in enumerate(self.indices, start=1):
            
            # Load current file & file -1
            img1 = cv2.imread(
                osp.join(curr_dir, 'dataset', self.d_name, f_name)
            )
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # type: ignore
            img2 = cv2.imread(
                osp.join(curr_dir, 'dataset', self.d_name, self.indices[i-1])
            )
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # type: ignore

            # Create SIFT object
            sift = cv2.SIFT_create()        # type: ignore
            kp1, des1 = sift.detectandCompute(gray1, None)
            kp2, des2 = sift.detectandCompute(gray2, None)

            # FLANN based Matcher
            FLANN_INDEX_KDTREE = self._cfg.exp.flann_index_kdtree
            flann = cv2.FlannBasedMatcher()

            # Find matches between the two images
            matches = flann.knnMatch(des1, des2, k=FLANN_INDEX_KDTREE)

            # Apply a ratio to select only good matches
            good_matches = [
                m for m, n in matches if m.distance < 0.12 
            ]



    def get_indices(self,) -> List[str]:
        """
        Get all the dataset indices for loading the dataset

        Returns:
            List[str] : list of indices containing the file name, shuffled
        """

        # Load based on dataset name
        if self.d_name != "fans":
            raise ValueError(f"Unknown dataset provided {self.d_name}")

        # Create list of indices for the dataset
        d_path = osp.join(curr_dir, "dataset", self.d_name)

        # Check number of items in dataset
        num_indices = len([name.strip() for name in os.listdir(d_path)])
        logger.debug(f"Number of items in dataset {num_indices}")

        # Get the indices
        indices = [name.strip() for name in os.listdir(d_path)]
        
        # Shuffle the indices to remove bias
        random.shuffle(indices)

        return indices

