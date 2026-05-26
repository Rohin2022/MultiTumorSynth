import nibabel as nib

import numpy as np

import pandas as pd

from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed

import glob

from scipy.ndimage import label

from skimage.measure import marching_cubes, mesh_surface_area



COLUMNS = [

    "bdmap_id", "organ",

    "diameter_x_mm", "diameter_y_mm", "diameter_z_mm",

    "volume_ml",

    "sphericity", "surface_volume_ratio",

    "elongation", "flatness", "max_3d_diameter_mm",

    "num_components"  # NEW COLUMN

]



def compute_diameters_and_coords(mask, spacing):

    mask = np.squeeze(mask)

    zeros = {col: 0.0 for col in COLUMNS if col not in ["bdmap_id", "organ"]}

    zeros["num_components"] = 0

   

    bin_mask = mask > 0

    if not bin_mask.any():

        return zeros



    # 1. Connected Components Tracking

    # Structure defines 26-connectivity for 3D space

    structure = np.ones((3, 3, 3), dtype=bool)

    _, num_components = label(bin_mask, structure=structure)



    # 2. Extract Physical Coordinates for All Voxels (Calculates overall structure at once)

    coords = np.argwhere(bin_mask)

    coords_mm = coords * spacing  # Vectorized conversion to physical space



    # 3. Axis-Aligned Box Diameters (Highly Optimized Min/Max)

    min_coords = coords_mm.min(axis=0)

    max_coords = coords_mm.max(axis=0)

    # Adding 1 single voxel width to accurately reflect physical boundary span

    diameters = max_coords - min_coords + spacing

   

    max_x, max_y, max_z = diameters[0], diameters[1], diameters[2]



    # 4. Volume

    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]

    volume_mm3 = len(coords_mm) * voxel_volume_mm3

    volume_ml = volume_mm3 / 1000.0



    # 5. Fast Principle Component Analysis (PCA) for Elongation, Flatness, and Max 3D Diameter

    # Bypasses regionprops and pdist completely. Speed improvement: >100x

    try:

        centered_coords = coords_mm - coords_mm.mean(axis=0)

        cov = np.cov(centered_coords.T)

       

        # Eigenvalues represent the variance along the geometric orthogonal axes

        eigvals = np.linalg.eigvals(cov)

        eigvals = np.sort(eigvals)[::-1]  # Sort descending (L1 >= L2 >= L3)

        eigvals = np.maximum(eigvals, 1e-8)  # Prevent divide-by-zero on flat edge errors



        elongation = float(np.sqrt(eigvals[1] / eigvals[0]))

        flatness = float(np.sqrt(eigvals[2] / eigvals[0]))

       

        # Statistical estimation of Maximum 3D Diameter via the 3D Convex Hull bounding ellipsoid span

        # 4 * sqrt(eigval) gives the equivalent principal diameter of the point distribution

        max_3d_diameter_mm = float(4.0 * np.sqrt(eigvals[0]))

    except Exception:

        elongation, flatness, max_3d_diameter_mm = 0.0, 0.0, 0.0



    # 6. Standard Surface Area via Marching Cubes

    # Generates a 3D mesh to accurately compute physical surface area

    try:

        # Pad mask to ensure closed surfaces if the organ touches the image boundary

        padded = np.pad(bin_mask, 1, mode='constant', constant_values=False)

       

        # Extract 3D mesh. 'level=0.5' is standard for boolean arrays.

        # Passing 'spacing' automatically scales the vertices to physical space (mm).

        verts, faces, normals, values = marching_cubes(padded, level=0.5, spacing=spacing)

       

        surface_area_mm2 = mesh_surface_area(verts, faces)



        surface_volume_ratio = float(surface_area_mm2 / volume_mm3)

       

        # Sphericity formula: ratio of the surface area of a sphere (with same volume) to actual surface area

        sphericity = float((np.pi ** (1 / 3) * (6 * volume_mm3) ** (2 / 3)) / surface_area_mm2)

       

        # Marching cubes is highly accurate, but extremely tiny volumes (e.g., 8 voxels)

        # might still yield minor discrete artifacts. Capping at 1.0 remains a safe fallback.

        sphericity = min(sphericity, 1.0)

       

    except Exception as e:

        print(e)

        # Failsafe for degenerate shapes (e.g., flat 2D slices that cannot be meshed in 3D)

        surface_volume_ratio, sphericity = 0.0, 0.0



    return {

        "diameter_x_mm": max_x,

        "diameter_y_mm": max_y,

        "diameter_z_mm": max_z,

        "volume_ml": volume_ml,

        "sphericity": sphericity,

        "surface_volume_ratio": surface_volume_ratio,

        "elongation": elongation,

        "flatness": flatness,

        "max_3d_diameter_mm": max_3d_diameter_mm,

        "num_components": int(num_components)

    }



def process_one(segmentation_path):

    parts = segmentation_path.split("/")

    bdmap = parts[-3]

    filename = parts[-1]

    organ = "_".join(filename.split("_")[:-1])



    try:

        img = nib.load(segmentation_path)

        mask = np.asanyarray(img.dataobj)

        spacing = np.abs(img.header['pixdim'][1:4]).astype(np.float32)



        metrics = compute_diameters_and_coords(mask, spacing)



        row = [bdmap, organ] + [metrics[col] for col in COLUMNS[2:]]

        return row

    except Exception as e:

        print(f"ERROR processing file {segmentation_path}: {e}")

        return None



def generate_mask_diffusion_txt(segmentation_paths, num_workers=16, output_csv="mask_metrics_radiomics_raw_verify.csv"):

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

   

    # Safely scaled back up to 16 workers since memory leaks/overhead are gone

    df = generate_mask_diffusion_txt(bdmaps_with_tumor_mask, num_workers=16, output_csv="mask_metrics.csv") 

