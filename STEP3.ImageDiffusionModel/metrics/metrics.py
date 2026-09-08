import logging

import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor

# Suppress verbose PyRadiomics logging
logging.getLogger("radiomics").setLevel(logging.ERROR)


class RadiomicsMetricsEvaluator:
    def __init__(self, spacing, bin_width=25):
        """
        Parameters
        ----------
        spacing : tuple(float, float, float)
            Physical voxel spacing (x, y, z) in mm.
        bin_width : float
            PyRadiomics intensity discretization bin width.
        """
        settings = {
            "geometryTolerance": 1e-4,
            "label": 1,
            "binWidth": bin_width,
        }

        self.extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        self.extractor.enableAllFeatures()
        # Uncomment if you also want wavelet/LoG features:
        # self.extractor.enableAllImageTypes()

        self.spacing = tuple(float(s) for s in spacing)

    def compute_radiomics(self, ct, tumor_mask):
        """
        Compute all PyRadiomics features for a tumor.

        Parameters
        ----------
        ct : ndarray or torch.Tensor
            3D CT volume in Hounsfield Units.
        tumor_mask : ndarray or torch.Tensor
            Binary tumor mask.

        Returns
        -------
        dict
            Dictionary mapping feature names to values.
            Diagnostic entries are removed.
        """

        # Convert tensors -> numpy
        if hasattr(ct, "detach"):
            ct = ct.detach().cpu().numpy()
        if hasattr(tumor_mask, "detach"):
            tumor_mask = tumor_mask.detach().cpu().numpy()

        ct = np.squeeze(ct).astype(np.float32)
        tumor_mask = (np.squeeze(tumor_mask) > 0).astype(np.uint8)

        if not np.any(tumor_mask):
            return {}

        # Convert to SimpleITK
        ct_sitk = sitk.GetImageFromArray(ct)
        mask_sitk = sitk.GetImageFromArray(tumor_mask)

        ct_sitk.SetSpacing(self.spacing)
        mask_sitk.SetSpacing(self.spacing)

        try:
            features = self.extractor.execute(ct_sitk, mask_sitk)

            result = {}
            for key, value in features.items():
                # Skip metadata/diagnostics
                if key.startswith("diagnostics_"):
                    continue

                try:
                    result[key] = float(value)
                except (TypeError, ValueError):
                    result[key] = value

            return result

        except Exception as e:
            print(f"PyRadiomics extraction failed: {e}")
            return {}