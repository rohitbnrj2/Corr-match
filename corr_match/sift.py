"""
SIFT (Scale-Invariant Feature Transform) implementation for feature matching 
to find correspondences between images. Using OpenCV.
"""
from __future__ import annotations

from typing import Any, List, Tuple, Sequence

import os
import os.path as osp
import random
import cv2

import numpy as np
from loguru import logger
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from pydantic import BaseModel

# Current Directory
curr_dir = osp.abspath(osp.dirname(__file__))


class Keypoints(BaseModel):
    """
    Keypoints for SIFT feature matching.
    """
    l: Sequence[cv2.KeyPoint]  # Keypoints for left image
    r: Sequence[cv2.KeyPoint]  # Keypoints for right image

    model_config = {"arbitrary_types_allowed": True}

class Descriptors(BaseModel):
    """
    Descriptors for SIFT feature matching.
    """
    l: np.ndarray  # Descriptors for left image
    r: np.ndarray  # Descriptors for right image

    model_config = {"arbitrary_types_allowed": True}


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

        # Set constants
        self.SIFT = cv2.SIFT_create()  # type: ignore

        FLANN_INDEX_KDTREE = self._cfg.exp.flann_index_kdtree
        TREES = self._cfg.exp.trees
        CHECKS = self._cfg.exp.flann_checks

        self.lowe_ratio = self._cfg.exp.lowe_ratio
        self.flann = cv2.FlannBasedMatcher(
            indexParams=dict(algorithm=FLANN_INDEX_KDTREE, trees=TREES),
            searchParams=dict(checks=CHECKS)
        )


    def __call__(self, ) -> None:
        """
        Run SIFT to get correspondence between pairs of images.
        """

        # Run the loop
        for _, f_name in enumerate(self.indices):

            # Load current file & random file
            logger.debug(f'Left Image Index: {f_name}')
            img1 = cv2.imread(
                osp.join(curr_dir, 'dataset', self.d_name, f_name)
            )
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # type: ignore

            idx = random.randint(0, len(self.indices)-1)
            logger.debug(f'Right Image Index: {self.indices[idx]}')
            img2 = cv2.imread(
                osp.join(curr_dir, 'dataset', self.d_name, self.indices[idx])
            )
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # type: ignore

            # Get SIFT keypoints and descriptors
            kps, des = self.sift_descriptors(gray1, gray2)

            # Find matches between the two images
            matches: Sequence[Sequence[cv2.DMatch]] = self.flann.knnMatch(
                des.l, des.r, k=2   # type: ignore
            )

            # Get Good Matches
            matches_mask = self.filter_matches(matches)

            # Draw Matches and log the result
            if img1 is not None and img2 is not None:
                self.draw_matches(
                    imgs=(img1, img2), kps=kps, matches=matches, matches_mask=matches_mask
                )
            
            else:
                raise ValueError("One or both images could not be loaded as they are None.")


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


    def sift_descriptors(self, img1: np.ndarray, img2:np.ndarray) -> Tuple[Keypoints, Descriptors]: 
        """
        Get the SIFT descriptors for a pair of images.

        Args:
            img1 (np.ndarray): The first image / left image
            img2 (np.ndarray): The second image / right image

        Returns:
            (kps, des): The SIFT keypoint and descriptor pairs for the images.
        """

        # Create SIFT object
        sift = cv2.SIFT_create()    # type: ignore

        # Get keypoints & descriptors for the images
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # Log the images to hydra
        log_images(
            img1=(img1, kp1), img2=(img2, kp2)
        )

        # Return the keypoints and descriptors
        return (
            Keypoints(l=kp1, r=kp2), Descriptors(l=des1, r=des2)
        )


    def filter_matches(self, matches: Sequence[Sequence[cv2.DMatch]]) -> List[List[int]]:
        """
        Filter matches using Lowe's ratio test.

        Args:
            matches (List[Tuple[cv2.DMatch, cv2.DMatch]]): The matches to filter.

        Returns:
            List[List[int]]: The filtered matches.
        """

        # Apply a ratio to select only good matches
        matches_mask = [[0,0] for i in range(len(matches))]

        # Ratio test as per Lowe's
        for j, (m,n) in enumerate(matches):
            if m.distance < self.lowe_ratio * n.distance:
                matches_mask[j] = [1,0]

        # Log the number of matches
        logger.info(
            f"Number of matches found: {sum(m[0] for m in matches_mask)}"
        )
        return matches_mask


    def draw_matches(self, 
        imgs: Tuple[np.ndarray, np.ndarray], kps: Keypoints, 
        matches: Sequence[Sequence[cv2.DMatch]], matches_mask: List[List[int]]
    ) -> None:
        """
        Draw matches between two images and log the result.

        Args:
            imgs (Tuple[np.ndarray, np.ndarray]): The images to draw matches on.
            kps (Keypoints): The keypoints for the images.
            matches (Sequence[Sequence[cv2.DMatch]]): The matches to draw.
            matches_mask (List[List[int]]): The mask indicating which matches to draw.

        Returns:
            None
        """

        # Draw Params
        draw_params = dict(
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            matchesMask=matches_mask,
            flags=0
        )

        # Draw matches
        log_img = cv2.drawMatchesKnn(
            img1=imgs[0], keypoints1=kps.l, 
            img2=imgs[1], keypoints2=kps.r, 
            matches1to2=matches, 
            outImg=None,  # type: ignore
            **draw_params # type: ignore
        )   # type: ignore

        # Log the image
        log_images(sift_matches=log_img)

COUNTER = 0
def log_images(**kwargs) -> None:
    """
    Log images to the Hydra output directory.
    
    This function saves images with SIFT features and matches to the current
    Hydra output directory, following Hydra best practices.
    
    Args:
        **kwargs: Keyword arguments containing images to log.
                 Expected format: sift_matches=image_array or similar
    """

    # Get the current Hydra output directory
    output_dir = HydraConfig.get().runtime.output_dir

    # Create images subdirectory if it doesn't exist
    images_dir = osp.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    for key, value in kwargs.items():

        if isinstance(value, np.ndarray):
            # Direct numpy array (image)
            global COUNTER
            filepath = osp.join(images_dir, f"{COUNTER:03d}_{key}.png")

            # Save the image
            try:
                cv2.imwrite(filepath, value)

            except Exception as e:
                logger.error(f"Failed to save image: {filepath}, Error: {e}")
                raise e

            COUNTER += 1

        elif isinstance(value, tuple) and len(value) == 2:
            # Tuple (image, keypoints)
            img, kp = value

            # Draw the keypoints on the image
            img_with_kp = cv2.drawKeypoints(
                img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )   # type: ignore

            filepath = osp.join(images_dir, f"{COUNTER:03d}_{key}.png")
            try:
                cv2.imwrite(filepath, img_with_kp)

            except Exception as e:
                logger.error(f"Failed to save image: {filepath}, Error: {e}")
                raise e

            COUNTER += 1

        else:
            logger.warning(
                f"Unsupported image format for key '{key}': {type(value)}"
            )
