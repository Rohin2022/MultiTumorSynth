import SimpleITK as sitk
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
from scipy.ndimage import label
from radiomics import featureextractor
import logging

# Silence PyRadiomics logger to avoid spamming the console during multiprocessing
logging.getLogger("radiomics").setLevel(logging.ERROR)

COLUMNS = [
    "bdmap_id", "organ",
    "diameter_x_mm", "diameter_y_mm", "diameter_z_mm",
    "volume_ml",
    "sphericity", "surface_volume_ratio",
    "elongation", "flatness", "max_3d_diameter_mm",
    "num_components"
]

def compute_metrics(mask_sitk):
    # 1. Convert to numpy array for fast bounding box and connected components
    mask_np = sitk.GetArrayFromImage(mask_sitk)
    bin_mask = mask_np > 0
    
    zeros = {col: 0.0 for col in COLUMNS if col not in ["bdmap_id", "organ"]}
    zeros["num_components"] = 0
    
    if not bin_mask.any():
        return zeros

    # 2. Connected Components Tracking
    structure = np.ones((3, 3, 3), dtype=bool)
    _, num_components = label(bin_mask, structure=structure)

    # 3. Axis-Aligned Box Diameters
    coords = np.argwhere(bin_mask) # Returns coordinates in (z, y, x) order
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    
    # SimpleITK spacing is (x, y, z). We reverse it to (z, y, x) to match numpy.
    spacing_xyz = mask_sitk.GetSpacing()
    spacing_zyx = np.array([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]])
    
    # Calculate diameters (adding 1 voxel to accurately reflect physical boundary span)
    diameters = (max_coords - min_coords + 1) * spacing_zyx
    max_z, max_y, max_x = diameters[0], diameters[1], diameters[2]

    # 4. Standardized Shape Features via PyRadiomics
    # PyRadiomics requires masks to be integer type (e.g., uint8) with distinct labels
    bin_mask_sitk = sitk.GetImageFromArray(bin_mask.astype(np.uint8))
    bin_mask_sitk.CopyInformation(mask_sitk) # Crucial: Transfers spacing/direction/origin

    # Configure the extractor for shape features only
    settings = {'geometryTolerance': 1e-4, 'label': 1}
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(shape=[
        'Elongation', 'Flatness', 'Maximum3DDiameter', 
        'MeshVolume', 'Sphericity', 'SurfaceVolumeRatio'
    ])

    try:
        # For shape features, we can pass the mask as both the image and the mask
        features = extractor.execute(bin_mask_sitk, bin_mask_sitk)
        
        elongation = float(features.get('original_shape_Elongation', 0.0))
        flatness = float(features.get('original_shape_Flatness', 0.0))
        max_3d_diameter = float(features.get('original_shape_Maximum3DDiameter', 0.0))
        volume_ml = float(features.get('original_shape_MeshVolume', 0.0)) / 1000.0
        sphericity = float(features.get('original_shape_Sphericity', 0.0))
        surface_volume_ratio = float(features.get('original_shape_SurfaceVolumeRatio', 0.0))
        
    except Exception as e:
        # Failsafe for degenerate shapes (e.g., single-voxel masks)
        elongation, flatness, max_3d_diameter = 0.0, 0.0, 0.0
        volume_ml, sphericity, surface_volume_ratio = 0.0, 0.0, 0.0

    return {
        "diameter_x_mm": max_x,
        "diameter_y_mm": max_y,
        "diameter_z_mm": max_z,
        "volume_ml": volume_ml,
        "sphericity": sphericity,
        "surface_volume_ratio": surface_volume_ratio,
        "elongation": elongation,
        "flatness": flatness,
        "max_3d_diameter_mm": max_3d_diameter,
        "num_components": int(num_components)
    }

def process_one(segmentation_path):
    parts = segmentation_path.split("/")
    bdmap = parts[-3]
    filename = parts[-1]
    organ = "_".join(filename.split("_")[:-1])

    try:
        # Load directly with SimpleITK
        mask_sitk = sitk.ReadImage(segmentation_path)
        
        metrics = compute_metrics(mask_sitk)
        
        row = [bdmap, organ] + [metrics[col] for col in COLUMNS[2:]]
        return row
    except Exception as e:
        print(f"ERROR processing file {segmentation_path}: {e}")
        return None

def generate_mask_diffusion_txt(segmentation_paths, num_workers=16, output_csv="mask_metrics.csv"):
    rows = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_one, p): p for p in segmentation_paths}

        for future in tqdm(as_completed(futures), total=len(segmentation_paths)):
            path = futures[future]
            try:
                result = future.result()
                if result is not None:
                    rows.append(result)

                if len(rows) % 500 == 0 and len(rows) > 0:
                    df_temp = pd.DataFrame(rows, columns=COLUMNS)
                    df_temp.to_csv(output_csv, index=False)
            except Exception as e:
                print(f"\nCRITICAL PROCESS FAILURE on file: {path}\nReason: {e}\n")

    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df.sort_values(["bdmap_id", "organ"]).reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} rows to {output_csv}")
    return df

if __name__ == '__main__':
    search_path = (
        "/projects/bodymaps/Data/"
        "radiologist_annotations_merlin_ucsf_atlas_multi_cancer/"
        "*/segmentations/*"
    )
    bdmaps_with_tumor_mask = glob.glob(search_path)
    
    df = generate_mask_diffusion_txt(bdmaps_with_tumor_mask, num_workers=16, output_csv="mask_metrics.csv")