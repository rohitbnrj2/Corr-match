"""
Generate Photon Binary Frames from RGB images, at 256x256 resolution
"""
from __future__ import annotations
from typing import Dict, Sequence, List, Tuple, Any

import os
import os.path as osp
from loguru import logger
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass, field
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

try:
    from einops import rearrange
    import imageio.v2 as io

except ImportError as e:
    logger.error("Install einops imageio to run the simulator")
    raise e

try:
    import torch
    import torchvision
    from torchvision import transforms
    from torch.utils.data import DataLoader, Dataset

except ImportError as e:
    logger.error(
        "Install PyTorch and Torchvision to run the simulator"
    )
    raise e

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'
plt.rcParams['image.cmap'] = 'gray'


@dataclass
class SimConfigs:
    """
    Configuration for the simulation.
    """

    # Dataset parameters
    dataset: str = field(default_factory=lambda: "Places365")
    split: str = "val"
    spatial_res: Tuple[int, int] = (256, 256)

    # DataLoader parameters
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False
    shuffle: bool = False
    drop_last: bool = True

    # Photon simulation parameters
    lambda_target: float = 1e-1  # Lambda(Mean) per exposure 
    pde: float = 0.6            # Photon Detection Efficiency
    dark_count: float = 2e2
    exposure_time: float = 1e-8


def sim_runner(cfgs: DictConfig) -> None:
    """
    Run the simulation.
    """

    logger.info("Running Data Generation Simulator...")
    sim = SimPhoton(cfgs)

    sim()


class SimDataset(Dataset):
    """
    Custom Dataset class to get RGB images for photon image simulations.
    """

    def __init__(self, cfg: SimConfigs) -> None:
        super().__init__()

        self.cfg = cfg
        self.dataset: Sequence = self.load_dataset()
        logger.debug(
            f"Split {self.cfg.split} | Dataset loaded with {len(self.dataset)} items"
        )


    def load_dataset(self) -> Sequence:
        """
        Load the dataset specified in the configuration.

        Returns:
            Sequence : Loaded dataset.
        """

        # Define the transform to resize and convert images to tensors
        transform = transforms.Compose([
            transforms.CenterCrop(self.cfg.spatial_res),
            transforms.ToTensor(),
        ])

        # Load the dataset
        try:
            dset_obj = getattr(torchvision.datasets, self.cfg.dataset)
            dataset = dset_obj(
                root=f'./data/{self.cfg.dataset.lower()}',
                split=self.cfg.split,
                transform=transform,
                download=True
            )
            logger.debug(
                f'Loaded {dset_obj} dataset with {len(dataset)} images.'
            )

        except (AttributeError, RuntimeError) as e:
            logger.error(
                f"Dataset {self.cfg.dataset} not found in torchvision.datasets"
            )
            raise e

        return dataset


    def __len__(self) -> int:
        return len(self.dataset)


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get an item from the dataset and return in a Hashmap.

        Args:
            idx (int) : Index of the item to retrieve.

        Returns:
            Dict : Dictionary containing item data.
        """
        elem = {}

        # Get the item from the dataset
        item = self.dataset[idx][0].numpy().astype(np.float64)
        item = rearrange(item, 'c h w -> h w c') * 255.0

        # Convert to Luminance image
        lum: np.ndarray = self.get_luminance(item)
        quant_lum: np.ndarray = self.convert_to_8bit(lum)

        # Convert to Photon binary frame
        photon_img, photon_counts = self.convert_to_photon(quant_lum)

        elem['image'] = item / 255.0
        elem['luminance'] = lum
        elem['quantized'] = quant_lum
        elem['photon'] = photon_img
        elem['counts'] = photon_counts
        elem['index'] = idx
        return elem


    def get_luminance(self, img: np.ndarray) -> np.ndarray:
        """
        Convert an sRGB image to luminance.
        
        Args:
            img (np.ndarray): Input sRGB image. [0, 1] range.

        Returns:
            np.ndarray: Luminance image. [0, 1] range.
        """

        # Normalize the image to [0, 1]
        if img.max() > 1.0:
            img = img / 255.0

        # Convert sRGB to RGB
        mask = img <= 0.04045
        img = np.where(
            mask, img / 12.92, ((img + 0.055) / 1.055) ** 2.4
        )

        # Convert linear RGB to luminance
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        lum = np.clip(lum, 0, 1).astype(np.float64)

        return lum


    def convert_to_8bit(self, img: np.ndarray) -> np.ndarray:
        """
        Convert luminance image to 8-bit quantized image.

        Args:
            img (np.ndarray): Input luminance image. [0, 1] range.

        Returns:
            np.ndarray: 8-bit quantized image. [0, 255] range.
        """

        if img.max() > 1.0:
            img = img / 255.0

        img *= 255.0
        quant_img = np.clip(img, 0, 255).astype(np.uint8)

        return quant_img


    def convert_to_photon(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert 8-bit image to photon binary frame.

        Args:
            img (np.ndarray): Input 8-bit image. [0, 255] range.
        Returns:
            np.ndarray: Photon binary frame. [0, 1] range.
        """

        if img.dtype != np.uint8:
            raise ValueError("Input image must be of type uint8")
        
        if img.max() > 255:
            raise ValueError(
                "Input image pixel values must be in the range [0, 255]"
            )

        # Normalize the image to [0, 1]
        img = img.astype(np.float64) / 255.0

        # Dark counts per exposure
        lambda_dark = self.cfg.dark_count * self.cfg.exposure_time
        logger.debug(f"Dark counts per exposure: {lambda_dark}")

        # Expected photons
        # Lambda_target is mean incident flux at lum(1.0) per exposure
        lambda_signal = img * self.cfg.lambda_target * self.cfg.pde
        logger.debug(f"Expected photons: {lambda_signal.mean():.2f}")

        # Photon detection probability
        total_lambda = np.clip(lambda_signal + lambda_dark, 0, None)

        # Detection as Poisson process
        photon_counts = np.random.poisson(total_lambda)
        photon_img = (photon_counts > 0).astype(np.uint8)

        return (photon_img, photon_counts)



class SimPhoton:
    """
    Generation of photon binary frames from RGB images.
    """

    def __init__(self, cfgs: DictConfig) -> None:

        self._cfg = cfgs.sim

        # Load the dataset
        self.dataset = SimDataset(self._cfg)
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=self._cfg.batch_size,
            num_workers=self._cfg.num_workers,
            shuffle=self._cfg.shuffle,
            drop_last=self._cfg.drop_last,
            pin_memory=self._cfg.pin_memory,
            persistent_workers=True if self._cfg.num_workers > 0 else False
        )
        logger.debug(
            f"DataLoader initialized with {len(self.dataloader)} batches."
        )


    def __call__(self,) -> None:
        """
        Simulate generation of photon binary frames from RGB images and log outputs.
        """

        pbar = tqdm(
            self.dataloader, total=len(self.dataloader), desc="Simulating Photon Frames"
        )

        for step, batch in enumerate(pbar):

            if batch['image'] is None:
                logger.warning(f"Batch {step} contains no images, skipping...")
                continue

            self.log_outputs(
                step=step, counts=batch['counts'], rgb=batch['image'], 
                luminance=batch['luminance'], quantized=batch['quantized'], 
                photon=batch['photon']
            )

        pbar.close()


    def log_outputs(self, **kwargs) -> None:
        """
        Log the outputs of the simulation step.

        Args:
            **kwargs: Keyword arguments containing step, rgb images, and photon images.
        """

        logger.info(
            f"Step {kwargs['step']}: Processed {len(kwargs['photon'])} images."
        )

        # Save directory for logging images
        save_dir = HydraConfig.get().runtime.output_dir
        os.makedirs(osp.join(save_dir, "photon_frames"), exist_ok=True)
        os.makedirs(osp.join(save_dir, "histograms"), exist_ok=True)

        # Setup figure and plots for images
        n = len(kwargs) - 2
        fig, axs = plt.subplots(self._cfg.batch_size // 8, n, figsize=(10, 10))
        if n == 1:
            axs = [[axs]]

        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        # Plot each image
        for i, (name, img) in enumerate(kwargs.items()):

            if name == 'step' or name == 'counts':
                continue

            # Subselect images to plot
            if isinstance(img[0], torch.Tensor):
                imgs = [img[s].numpy() for s in range(0, img.shape[0], 8)]

            else:
                imgs = [img[s] for s in range(0, len(img), 8)]

            # Plot each image in the batch
            for j, im in enumerate(imgs):

                if im.ndim == 2:
                    axs[j][i-2].imshow(im, cmap='gray')
                else:
                    axs[j][i-2].imshow(im)

                axs[j][i-2].axis('off')
                axs[j][i-2].set_title(f"{name.capitalize()}", fontsize=10)

        # Save the figure with the combined images
        plt.tight_layout()
        plt.savefig(
            osp.join(save_dir, "photon_frames", f"step_{kwargs['step']:04d}.png"),
            bbox_inches='tight', dpi=300
        )
        plt.close(fig)

        # Create overlapped histogram for luminance and photon images
        if 'luminance' in kwargs and 'photon' in kwargs:
            self.create_histogram_visualization(
                luminance_images=kwargs['luminance'],
                photon_counts=kwargs['counts'],
                step=kwargs['step'],
                save_dir=save_dir
            )
 
        logger.debug(f"Saved images for step {kwargs['step']} in {save_dir}/photon_frames")


    def create_histogram_visualization(self, 
            luminance_images: List[torch.Tensor], photon_counts: List[torch.Tensor], 
            step: int, save_dir: str
        ) -> None:
        """
        Create overlapped histograms for luminance and photon images.
        
        Args:
            luminance_images: Batch of luminance images
            photon_counts: Batch of photon counts
            step: Current step number
            save_dir: Directory to save the histogram
        """
        
        # Create figure for histogram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
        
        # Convert tensors to numpy if needed
        if isinstance(luminance_images[0], torch.Tensor):

            norm_lum = [img / 255.0 if img.max() > 1.0 else img for img in luminance_images]
            lum_data = [img.numpy().flatten() for img in norm_lum]

            photon_data = [img.numpy().flatten() for img in photon_counts]

        else:

            norm_lum = [img / 255.0 if img.max() > 1.0 else img for img in luminance_images]
            lum_data = [img.flatten() for img in norm_lum]

            photon_data = [img.flatten() for img in photon_counts]


        # Concatenate all image data for the batch
        all_luminance = rearrange(np.stack(lum_data), 'b v -> (b v)')
        all_photon = rearrange(np.stack(photon_data), 'b v -> (b v)')

        # Plot overlapped histograms
        ax1.hist(all_luminance, bins=50, alpha=0.7, color='blue', 
            label='Luminance (μ={:.3f}, σ={:.3f})'.format(all_luminance.mean(), all_luminance.std()), 
            density=True, edgecolor='black', linewidth=0.5)

        ax1.set_xlim(0, 1)
        ax1.set_xticks(np.arange(0, 1.1, 0.2))
        ax1.set_xlabel('Pixel Intensity')
        ax1.set_ylabel('Probability Density (PDF)')
        ax1.set_title(f'Histograms - Step {step}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # ------------------ Photon counts (log PDF so small probs show) ------------------
        # create histogram manually so we can inspect zero bins
        bins = np.arange(0, 11, 1)  # 0..10 integer photon counts (adjust if needed)
        counts, bin_edges = np.histogram(all_photon, bins=bins, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        widths = np.diff(bin_edges)

        # plot bars (heights are true densities)
        ax2.bar(bin_centers, counts, width=widths, align='center',
                color='salmon', alpha=0.85, edgecolor='black', linewidth=0.4,
                label=f'Photon Counts (μ={all_photon.mean():.3f}, σ={all_photon.std():.3f})')

        ax2.set_xlim(bin_edges[0], bin_edges[-1])
        ax2.set_xlabel('Photon Counts')
        ax2.set_ylabel('Probability Density (PMF)')
        ax2.set_title(f'Photon Counts Histogram - Step {step:02d}')
        ax2.legend(loc='upper right')
        ax2.grid(alpha=0.25)

        # --- switch to log scale on y-axis so small densities are visible ---
        ax2.set_yscale('log')

        plt.tight_layout()
        plt.savefig(
            osp.join(save_dir, "histograms", f"histogram_step_{step:04d}.png"),
            bbox_inches='tight', dpi=300
        )
        plt.close(fig)
        logger.debug(f"Saved histogram for step {step} in {save_dir}/histograms")
