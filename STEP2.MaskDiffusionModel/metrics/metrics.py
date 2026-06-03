import SimpleITK as sitk
import numpy as np
from scipy.ndimage import label
from radiomics import featureextractor
import logging

# Suppress verbose pyradiomics logging
logging.getLogger("radiomics").setLevel(logging.ERROR)

class RadiomicsMetricsEvaluator:
    def __init__(self):
        """
        Initializes the PyRadiomics feature extractor.
        Done once during instantiation to save overhead.
        """
        settings = {'geometryTolerance': 1e-4, 'label': 1}
        self.extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        self.extractor.disableAllFeatures()
        
        # Enabled features matching the 10 target conditioners
        self.extractor.enableFeaturesByName(shape=[
            'Elongation', 'Flatness', 'Maximum3DDiameter',
            'MeshVolume', 'Sphericity', 'SurfaceVolumeRatio',
            'MajorAxisLength', 'MinorAxisLength', 'LeastAxisLength'
        ])

        # The exact order of your 10 continuous features
        self.columns = [
            "major_axis_mm", "minor_axis_mm", "least_axis_mm",
            "volume_ml", "sphericity", "surface_volume_ratio",
            "elongation", "flatness", "max_3d_diameter_mm",
            "num_components"
        ]

    def compute(self, mask, spacing):
        """
        Computes volume, axes, and shape features using PyRadiomics.
        
        Args:
            mask: 3D binary mask (numpy array or torch tensor)
            spacing: Tuple/list of physical voxel spacing (x, y, z)
        
        Returns:
            Dictionary mapping feature names to their computed values.
        """
        if hasattr(mask, "numpy"):
            mask = mask.cpu().numpy()

        mask = np.squeeze(mask)
        bin_mask = (mask > 0).astype(np.uint8)

        # Initialize defaults
        metrics = {col: 0.0 for col in self.columns}
        metrics["num_components"] = 0

        if not bin_mask.any():
            return metrics

        # 1. Connected Components Tracking
        structure = np.ones((3, 3, 3), dtype=bool)
        _, num_components = label(bin_mask, structure=structure)
        metrics["num_components"] = int(num_components)

        # 2. PyRadiomics Feature Extraction
        bin_mask_sitk = sitk.GetImageFromArray(bin_mask)
        
        # Ensure spacing is passed as standard Python floats (SimpleITK requirement)
        bin_mask_sitk.SetSpacing([float(s) for s in spacing])

        try:
            features = self.extractor.execute(bin_mask_sitk, bin_mask_sitk)

            metrics["elongation"]           = float(features.get('original_shape_Elongation', 0.0))
            metrics["flatness"]             = float(features.get('original_shape_Flatness', 0.0))
            metrics["max_3d_diameter_mm"]   = float(features.get('original_shape_Maximum3DDiameter', 0.0))
            metrics["volume_ml"]            = float(features.get('original_shape_MeshVolume', 0.0)) / 1000.0
            metrics["sphericity"]           = float(features.get('original_shape_Sphericity', 0.0))
            metrics["surface_volume_ratio"] = float(features.get('original_shape_SurfaceVolumeRatio', 0.0))
            metrics["major_axis_mm"]        = float(features.get('original_shape_MajorAxisLength', 0.0))
            metrics["minor_axis_mm"]        = float(features.get('original_shape_MinorAxisLength', 0.0))
            metrics["least_axis_mm"]        = float(features.get('original_shape_LeastAxisLength', 0.0))

        except Exception as e:
            print(f"PyRadiomics extraction failed on inference sample: {e}")

        return metrics