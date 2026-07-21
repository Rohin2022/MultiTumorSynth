import pickle
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# GMM radiomics sampling
# --------------------------------------------------------------------------- #
def load_gmm_bank(path):
    """
    Loads a per-organ GMM + preprocessing bundle, pickled as:

        {
          organ_name: {
             "gmm": GaussianMixture,
             "power_transformer": PowerTransformer,
             "scaler": StandardScaler,
             "pca": PCA,
          },
          ...
        }
    """
    with open(path, "rb") as f:
        bank = pickle.load(f)
    return bank


def load_gmm_banks_from_config(cfg):
    """Convenience loader for both GMM banks using cfg.paths.mask_gmm_bank /
    cfg.paths.tumor_gmm_bank."""
    mask_gmm_bank = load_gmm_bank(cfg.paths.mask_gmm_bank)
    tumor_gmm_bank = load_gmm_bank(cfg.paths.tumor_gmm_bank)
    return mask_gmm_bank, tumor_gmm_bank


def _sample_from_gmm_bundle(bundle, n_samples=1):
    """Sample n_samples from a fitted GMM and invert PCA -> scaler -> power transform
    to map back into the original radiomics feature space."""
    gmm = bundle["gmm"]
    pca = bundle["pca"]
    scaler = bundle["scaler"]
    pt = bundle["power_transformer"]

    z, _ = gmm.sample(n_samples)                # samples live in PCA space
    x_scaled = pca.inverse_transform(z)         # -> standardized space
    x_std = scaler.inverse_transform(x_scaled)  # -> power-transformed space
    x = pt.inverse_transform(x_std)             # -> original radiomics units
    return x  # (n_samples, n_features)


def synthesize_organ_radiomics(
    gmm_bank_path,
    organ,
    num_samples,
    reference_sample=None,
    jitter_scale=0.05,
):
    """
    Loads a GMM bank from `gmm_bank_path` and returns `num_samples` synthetic
    radiomics vectors for `organ`, keyed by the feature names stored in the
    bundle at training time.

    Normal mode (reference_sample=None): draws fresh samples from the fitted
    GMM, as before.

    Reference mode (reference_sample given): instead of sampling from the
    GMM, generates variants by jittering around a real feature vector you
    pulled from your dataset (e.g. a known large tumor). This preserves the
    exact feature correlations of that real sample while giving some
    diversity across the num_samples outputs.

    Args:
        gmm_bank_path: path to a pickled GMM bank (see load_gmm_bank).
        organ: organ name key into the bank.
        num_samples: number of synthetic samples to draw/generate.
        reference_sample: optional dict[str, float] mapping feature name ->
            value, e.g. pulled directly from a real large-tumor case in your
            dataset. Must contain (at least) all keys in the bundle's
            feature_names; extra keys are ignored.
        jitter_scale: float in [0, ~0.3]. Fraction of each feature's fitted
            GMM std to use as jitter noise stddev when reference_sample is
            given. 0 = return the reference sample repeated exactly
            (no diversity). Ignored if reference_sample is None.

    Returns:
        dict[str, np.ndarray] mapping each feature name -> array of shape
        (num_samples,) of synthesized values.
    """
    bank = load_gmm_bank(gmm_bank_path)

    if organ not in bank:
        raise KeyError(
            f"Organ '{organ}' not found in GMM bank at {gmm_bank_path}. "
            f"Available organs: {list(bank.keys())}"
        )

    bundle = bank[organ]

    if "feature_names" not in bundle:
        raise KeyError(
            f"Bundle for organ '{organ}' has no 'feature_names' key. "
            f"Re-save the GMM bank with feature_names included."
        )

    feature_names = bundle["feature_names"]

    if reference_sample is not None:
        missing = [f for f in feature_names if f not in reference_sample]
        if missing:
            raise KeyError(
                f"reference_sample is missing required features for organ "
                f"'{organ}': {missing}"
            )

        ref_vec = np.array([reference_sample[f] for f in feature_names], dtype=float)

        if jitter_scale > 0:
            # Use the GMM's own fitted spread (in original units) to scale
            # jitter per-feature, so noise is meaningful for each feature's
            # natural range. Draw a pilot batch purely to estimate std;
            # cheap and avoids needing separately stored per-feature stats.
            pilot = _sample_from_gmm_bundle(bundle, n_samples=max(500, num_samples * 10))
            feature_std = pilot.std(axis=0)

            noise = np.random.normal(
                loc=0.0,
                scale=jitter_scale * feature_std,
                size=(num_samples, len(feature_names)),
            )
            x = ref_vec[None, :] + noise
        else:
            x = np.tile(ref_vec, (num_samples, 1))

        return {feat: x[:, i] for i, feat in enumerate(feature_names)}

    # --- original GMM-sampling path ---
    x = _sample_from_gmm_bundle(bundle, n_samples=num_samples)  # (n_samples, n_features)

    if x.shape[1] != len(feature_names):
        raise ValueError(
            f"Mismatch between sampled feature dim ({x.shape[1]}) and "
            f"len(feature_names) ({len(feature_names)}) stored in bundle "
            f"for organ '{organ}'."
        )

    return {feat: x[:, i] for i, feat in enumerate(feature_names)}


def split_tumor_shape_features(sampled_features, tumor_keys, shape_keys):
    """
    Splits the output of `synthesize_organ_radiomics` (a dict keyed by all
    trained-on features) into two sub-dicts: tumor radiomics features and
    shape/mask radiomics features.

    Args:
        sampled_features: dict[str, np.ndarray] as returned by
            synthesize_organ_radiomics (or _sample_from_gmm_bundle + zip).
        tumor_keys: list of feature names to extract as "tumor" features.
        shape_keys: list of feature names to extract as "shape" features.

    Returns:
        (tumor_features, shape_features): two dicts, each a subset of
        sampled_features restricted to the requested keys.

    Raises:
        KeyError if a requested key isn't present in sampled_features.
    """
    missing_tumor = [k for k in tumor_keys if k not in sampled_features]
    missing_shape = [k for k in shape_keys if k not in sampled_features]
    if missing_tumor:
        raise KeyError(
            f"tumor_keys not found in sampled_features: {missing_tumor}")
    if missing_shape:
        raise KeyError(
            f"shape_keys not found in sampled_features: {missing_shape}")

    tumor_features = {k: sampled_features[k] for k in tumor_keys}
    shape_features = {k: sampled_features[k] for k in shape_keys}

    return tumor_features, shape_features


def apply_normalization(radiomics, tumor_norm_stats, mask_norm_stats):

    for key, stats in tumor_norm_stats.items():
        if key in radiomics["tumor_radiomics"]:
            radiomics["tumor_radiomics"][key] = (radiomics["tumor_radiomics"][key] -
                                                 stats["mean"]) / stats["std"]

            # Apply the normalization (using loaded or newly generated stats)
    for key, stats in mask_norm_stats.items():
        if key in radiomics["mask_radiomics"]:
            radiomics["mask_radiomics"][key] = (radiomics["mask_radiomics"][key] -
                                                stats["mean"]) / stats["std"]
    return radiomics
