"""Data Preparation and Quality — SmartVal AI

Implements outlier detection, missing value analysis, imputation, class imbalance 
handling, and data validation. These are the core data quality skills that prevent
production ML failures.

Learning Goals:
- Detect outliers using IQR and Z-score methods
- Analyze missing value patterns (MCAR, MAR, MNAR)
- Compare imputation strategies (mean, median, KNN)
- Handle class imbalance with SMOTE
- Validate data quality with PSI and KS tests

Prerequisites: Read notes/01-ml/01_regression/ch00_data_prep/README.md
Time Estimate: 3-4 hours
"""

import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

logger = logging.getLogger("smartval.data_prep")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: OUTLIER DETECTION (30 minutes)
# ═══════════════════════════════════════════════════════════════════════════


def detect_outliers_iqr(
    df: pd.DataFrame, 
    column: str, 
    multiplier: float = 1.5
) -> pd.DataFrame:
    """Detect outliers using the IQR (Interquartile Range) method.
    
    The IQR method is robust to extreme values because it uses percentiles
    rather than mean/std. It flags values outside [Q1 - k*IQR, Q3 + k*IQR]
    where k is typically 1.5 (Tukey's original convention from 1977).
    
    Algorithm:
    1. Compute Q1 (25th percentile) and Q3 (75th percentile)
    2. Compute IQR = Q3 - Q1
    3. Define fences: lower = Q1 - multiplier*IQR, upper = Q3 + multiplier*IQR
    4. Flag any value outside [lower, upper] as an outlier
    
    Args:
        df: Input dataframe
        column: Column name to check for outliers
        multiplier: IQR multiplier (default 1.5 retains ~99.3% of normal data)
    
    Returns:
        Dataframe containing only the outlier rows
        
    Example:
        >>> outliers = detect_outliers_iqr(df, 'HouseAge', multiplier=1.5)
        >>> print(f"Found {len(outliers)} outliers in HouseAge")
    
    TODO #1: Implement IQR outlier detection (10 minutes)
    Hints:
    - Use df[column].quantile(0.25) and df[column].quantile(0.75)
    - Calculate IQR = Q3 - Q1
    - Define lower_fence and upper_fence using the formula above
    - Return df[(df[column] < lower_fence) | (df[column] > upper_fence)]
    """
    # TODO: Compute Q1 and Q3
    Q1 = None  # Replace with actual computation
    Q3 = None  # Replace with actual computation
    
    # TODO: Compute IQR
    IQR = None  # Replace: Q3 - Q1
    
    # TODO: Define fences
    lower_fence = None  # Replace: Q1 - multiplier * IQR
    upper_fence = None  # Replace: Q3 + multiplier * IQR
    
    # TODO: Return outlier rows
    outliers = None  # Replace with boolean mask filter
    
    logger.info(
        f"IQR outlier detection on {column}: "
        f"Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}, "
        f"fences=[{lower_fence:.2f}, {upper_fence:.2f}], "
        f"outliers={len(outliers) if outliers is not None else 0}"
    )
    
    return outliers


def detect_outliers_zscore(
    df: pd.DataFrame, 
    column: str, 
    threshold: float = 3.0
) -> pd.DataFrame:
    """Detect outliers using Z-score (standardized distance from mean).
    
    Z-score measures how many standard deviations a value is from the mean:
    Z = (x - mean) / std
    
    Convention: |Z| > 3 is an outlier (retains 99.73% of normal data).
    
    ⚠️ WARNING: Z-score assumes roughly normal data and suffers from 
    "masking" where extreme outliers inflate the mean and std, making
    themselves harder to detect. Prefer IQR for most cases.
    
    Args:
        df: Input dataframe
        column: Column name to check
        threshold: Z-score threshold (default 3.0)
    
    Returns:
        Dataframe containing only the outlier rows
        
    TODO #2: Implement Z-score outlier detection (5 minutes)
    Hints:
    - Use stats.zscore(df[column]) to compute Z-scores
    - Flag rows where np.abs(z_scores) > threshold
    - Return the filtered dataframe
    """
    # TODO: Compute Z-scores for the column
    z_scores = None  # Replace with actual Z-score computation
    
    # TODO: Find rows where |Z| > threshold
    outliers = None  # Replace with boolean mask filter
    
    logger.info(
        f"Z-score outlier detection on {column}: "
        f"threshold={threshold}, outliers={len(outliers) if outliers is not None else 0}"
    )
    
    return outliers


def remove_outliers(
    df: pd.DataFrame, 
    outlier_indices: pd.Index
) -> pd.DataFrame:
    """Remove outlier rows from dataframe.
    
    Args:
        df: Input dataframe
        outlier_indices: Index of rows to remove
    
    Returns:
        Cleaned dataframe with outliers removed
        
    TODO #3: Implement outlier removal (2 minutes)
    Hint: Use df.drop(outlier_indices) to remove rows by index
    """
    # TODO: Drop outlier rows and return cleaned dataframe
    df_clean = None  # Replace with actual drop operation
    
    logger.info(f"Removed {len(outlier_indices)} outlier rows")
    
    return df_clean


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: MISSING VALUE ANALYSIS (30 minutes)
# ═══════════════════════════════════════════════════════════════════════════


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Generate missing value summary report.
    
    For each column, compute:
    - Count of missing values
    - Percentage of missing values
    - Data type
    
    Args:
        df: Input dataframe
    
    Returns:
        DataFrame with columns: [column_name, missing_count, missing_pct, dtype]
        
    TODO #4: Implement missing value analysis (10 minutes)
    Hints:
    - Use df.isnull().sum() to count missing values per column
    - Calculate percentage: (count / len(df)) * 100
    - Create a summary dataframe with the required columns
    - Filter to show only columns with missing_count > 0
    """
    # TODO: Count missing values per column
    missing_counts = None  # Replace with actual computation
    
    # TODO: Calculate missing percentages
    missing_pct = None  # Replace: (missing_counts / len(df)) * 100
    
    # TODO: Create summary dataframe
    summary = None  # Replace with pd.DataFrame creation
    
    # TODO: Filter to show only columns with missing values
    summary_filtered = None  # Replace with filter
    
    logger.info(f"Missing value analysis: {len(summary_filtered)} columns have missing data")
    
    return summary_filtered


def check_missing_pattern(
    df: pd.DataFrame, 
    target_col: str, 
    feature_col: str
) -> str:
    """Determine if missingness is MCAR, MAR, or MNAR.
    
    Missing data patterns:
    - MCAR (Missing Completely At Random): missingness independent of all variables
    - MAR (Missing At Random): missingness depends on observed variables
    - MNAR (Missing Not At Random): missingness depends on the missing value itself
    
    Test: Compare target mean for rows where feature is missing vs. not missing.
    If means differ significantly, the pattern is likely MAR.
    
    Args:
        df: Input dataframe
        target_col: Target variable column name
        feature_col: Feature column to test for missingness pattern
    
    Returns:
        String: "MCAR", "MAR", or "MNAR" (simplified heuristic)
        
    TODO #5: Implement missing pattern detection (10 minutes)
    Hints:
    - Split data into: rows where feature_col is missing vs. not missing
    - Compute mean of target_col for both groups
    - Use scipy.stats.ttest_ind() to test if means differ significantly
    - If p-value < 0.05, pattern is likely MAR; otherwise MCAR
    - (MNAR requires domain knowledge; we simplify to MCAR or MAR)
    """
    # TODO: Split into missing and non-missing groups
    missing_mask = None  # Replace with boolean mask: df[feature_col].isnull()
    
    target_missing = None  # Replace: df.loc[missing_mask, target_col]
    target_not_missing = None  # Replace: df.loc[~missing_mask, target_col]
    
    # TODO: Perform t-test
    t_stat, p_value = None, None  # Replace with stats.ttest_ind()
    
    # TODO: Determine pattern based on p-value
    if p_value is None:
        pattern = "UNKNOWN"
    elif p_value < 0.05:
        pattern = "MAR"  # Missingness depends on target (observed variable)
    else:
        pattern = "MCAR"  # Missingness appears random
    
    logger.info(
        f"Missing pattern for {feature_col}: {pattern} "
        f"(t={t_stat:.2f}, p={p_value:.4f})" if t_stat else f"{pattern}"
    )
    
    return pattern


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: IMPUTATION STRATEGIES (45 minutes)
# ═══════════════════════════════════════════════════════════════════════════


def impute_mean(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    columns: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Impute missing values with column mean (computed on train set only).
    
    ⚠️ CRITICAL: Fit imputer on train set, transform both train and test.
    This prevents data leakage (test statistics influencing training).
    
    Args:
        X_train: Training features
        X_test: Test features
        columns: Columns to impute
    
    Returns:
        Tuple of (X_train_imputed, X_test_imputed)
        
    TODO #6: Implement mean imputation (10 minutes)
    Hints:
    - Use SimpleImputer(strategy='mean')
    - Fit on X_train[columns]
    - Transform both X_train[columns] and X_test[columns]
    - Return copies of the dataframes with imputed values
    """
    # TODO: Create mean imputer
    imputer = None  # Replace with SimpleImputer(strategy='mean')
    
    # TODO: Fit on training data only
    # imputer.fit(X_train[columns])
    
    # TODO: Transform both train and test
    X_train_imputed = X_train.copy()
    X_test_imputed = X_test.copy()
    
    # X_train_imputed[columns] = imputer.transform(X_train[columns])
    # X_test_imputed[columns] = imputer.transform(X_test[columns])
    
    logger.info(f"Mean imputation applied to columns: {columns}")
    
    return X_train_imputed, X_test_imputed


def impute_median(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    columns: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Impute missing values with column median (robust to outliers).
    
    Median imputation is preferred over mean when:
    - Distribution is skewed
    - Outliers are present
    - You want a more "typical" value
    
    Args:
        X_train: Training features
        X_test: Test features
        columns: Columns to impute
    
    Returns:
        Tuple of (X_train_imputed, X_test_imputed)
        
    TODO #7: Implement median imputation (5 minutes)
    Hints:
    - Use SimpleImputer(strategy='median')
    - Follow the same fit-on-train, transform-both pattern as mean imputation
    """
    # TODO: Implement median imputation (similar to mean)
    imputer = None
    
    X_train_imputed = X_train.copy()
    X_test_imputed = X_test.copy()
    
    # Add your implementation here
    
    logger.info(f"Median imputation applied to columns: {columns}")
    
    return X_train_imputed, X_test_imputed


def impute_knn(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    columns: List[str], 
    n_neighbors: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Impute missing values using K-Nearest Neighbors.
    
    KNN imputation uses similar rows to estimate missing values:
    1. Find k nearest neighbors (by Euclidean distance) with non-missing values
    2. Average their values to fill the gap
    
    More sophisticated than mean/median but:
    - Requires scaled features (distance-based)
    - Slower to compute
    - Can capture multivariate patterns
    
    Args:
        X_train: Training features
        X_test: Test features
        columns: Columns to impute
        n_neighbors: Number of neighbors to use (default 5)
    
    Returns:
        Tuple of (X_train_imputed, X_test_imputed)
        
    TODO #8: Implement KNN imputation (15 minutes)
    Hints:
    - Scale features first using StandardScaler (KNN is distance-based)
    - Use KNNImputer(n_neighbors=n_neighbors)
    - Fit on scaled train data, transform both train and test
    - Inverse transform to get back to original scale
    """
    # TODO: Create scaler and fit on training data
    scaler = None  # Replace with StandardScaler()
    
    # TODO: Scale train and test
    X_train_scaled = None  # scaler.fit_transform(X_train)
    X_test_scaled = None  # scaler.transform(X_test)
    
    # TODO: Create KNN imputer
    imputer = None  # Replace with KNNImputer(n_neighbors=n_neighbors)
    
    # TODO: Fit on scaled training data and transform both
    X_train_imputed_scaled = None  # imputer.fit_transform(X_train_scaled)
    X_test_imputed_scaled = None  # imputer.transform(X_test_scaled)
    
    # TODO: Inverse transform to original scale
    X_train_imputed = None  # scaler.inverse_transform(X_train_imputed_scaled)
    X_test_imputed = None  # scaler.inverse_transform(X_test_imputed_scaled)
    
    # TODO: Convert back to DataFrames
    X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns, index=X_train.index)
    X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_test.columns, index=X_test.index)
    
    logger.info(f"KNN imputation (k={n_neighbors}) applied to columns: {columns}")
    
    return X_train_imputed, X_test_imputed


def compare_imputation_strategies(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    columns: List[str],
    model_class
) -> Dict[str, float]:
    """Compare MAE for different imputation strategies.
    
    Trains a model with each imputation method and reports test MAE.
    This is how you decide which strategy works best for your data.
    
    Args:
        X_train, X_test: Features
        y_train, y_test: Targets
        columns: Columns to impute
        model_class: Sklearn model class (e.g., LinearRegression)
    
    Returns:
        Dict mapping strategy name to test MAE
        
    TODO #9: Implement imputation comparison (15 minutes)
    Hints:
    - Call each imputation function (mean, median, knn)
    - Train model on imputed train set
    - Predict on imputed test set
    - Compute MAE using sklearn.metrics.mean_absolute_error
    - Return dict like {"mean": 52.1, "median": 51.8, "knn": 50.2}
    """
    from sklearn.metrics import mean_absolute_error
    
    results = {}
    
    # TODO: Test mean imputation
    X_train_mean, X_test_mean = None, None  # Replace with impute_mean()
    # Train model, predict, compute MAE, store in results["mean"]
    
    # TODO: Test median imputation
    X_train_median, X_test_median = None, None  # Replace with impute_median()
    # Train model, predict, compute MAE, store in results["median"]
    
    # TODO: Test KNN imputation
    X_train_knn, X_test_knn = None, None  # Replace with impute_knn()
    # Train model, predict, compute MAE, store in results["knn"]
    
    logger.info(f"Imputation comparison results: {results}")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: CLASS IMBALANCE (30 minutes)
# ═══════════════════════════════════════════════════════════════════════════


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: str = 'auto',
    k_neighbors: int = 5
) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE (Synthetic Minority Over-sampling Technique).
    
    SMOTE addresses class imbalance by creating synthetic samples:
    1. For each minority class sample, find k nearest neighbors
    2. Randomly select one neighbor
    3. Create synthetic sample: x_new = x + λ(x_neighbor - x), λ ∈ [0, 1]
    
    ⚠️ CRITICAL: Apply SMOTE AFTER train/test split to prevent data leakage.
    Synthetic samples are interpolations of real data — if you SMOTE before
    splitting, your test set contains "echoes" of your training data.
    
    Args:
        X_train: Training features
        y_train: Training labels
        sampling_strategy: 'auto' (balance to majority), 'minority', or dict
        k_neighbors: Number of nearest neighbors for synthesis
    
    Returns:
        Tuple of (X_train_resampled, y_train_resampled)
        
    TODO #10: Implement SMOTE resampling (15 minutes)
    Hints:
    - Use SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=42)
    - Call fit_resample(X_train, y_train)
    - Convert result back to DataFrame/Series with proper column names
    - Log class distribution before and after
    """
    # TODO: Log original class distribution
    original_dist = None  # y_train.value_counts()
    
    # TODO: Create SMOTE instance
    smote = None  # Replace with SMOTE(...)
    
    # TODO: Resample
    X_resampled, y_resampled = None, None  # smote.fit_resample(X_train, y_train)
    
    # TODO: Convert back to DataFrame/Series
    X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    y_resampled = pd.Series(y_resampled, name=y_train.name)
    
    # TODO: Log new class distribution
    new_dist = None  # y_resampled.value_counts()
    
    logger.info(
        f"SMOTE resampling: {len(X_train)} → {len(X_resampled)} samples"
    )
    
    return X_resampled, y_resampled


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: DATA VALIDATION (30 minutes)
# ═══════════════════════════════════════════════════════════════════════════


def compute_psi(
    expected: pd.Series,
    actual: pd.Series,
    bins: int = 10
) -> float:
    """Compute Population Stability Index (PSI).
    
    PSI measures how much a distribution has shifted between two datasets
    (typically training vs. production). Used extensively in credit risk.
    
    Formula:
    PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
    
    Interpretation:
    - PSI < 0.1: No significant shift
    - 0.1 ≤ PSI < 0.2: Moderate shift (investigate)
    - PSI ≥ 0.2: Significant shift (retrain model)
    
    Args:
        expected: Reference distribution (e.g., training data)
        actual: Current distribution (e.g., production data)
        bins: Number of bins for bucketing
    
    Returns:
        PSI score (float)
        
    TODO #11: Implement PSI calculation (15 minutes)
    Hints:
    - Use pd.cut() to bin both distributions into equal-width buckets
    - Compute percentage in each bin for expected and actual
    - Add small epsilon (1e-4) to avoid log(0)
    - PSI = sum((actual_pct - expected_pct) * log(actual_pct / expected_pct))
    """
    # TODO: Define bin edges based on expected distribution
    min_val, max_val = None, None  # min/max of expected and actual combined
    bin_edges = None  # np.linspace(min_val, max_val, bins + 1)
    
    # TODO: Bin both distributions
    expected_binned = None  # pd.cut(expected, bins=bin_edges)
    actual_binned = None  # pd.cut(actual, bins=bin_edges)
    
    # TODO: Compute percentages in each bin
    expected_pct = None  # expected_binned.value_counts(normalize=True, sort=False)
    actual_pct = None  # actual_binned.value_counts(normalize=True, sort=False)
    
    # TODO: Add epsilon to avoid log(0)
    epsilon = 1e-4
    expected_pct += epsilon
    actual_pct += epsilon
    
    # TODO: Compute PSI
    psi = None  # sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    
    logger.info(f"PSI = {psi:.4f}")
    
    return psi


def ks_test_drift(
    expected: pd.Series,
    actual: pd.Series
) -> Tuple[float, float]:
    """Perform Kolmogorov-Smirnov test for distribution drift.
    
    KS test measures the maximum distance between two cumulative distribution
    functions (CDFs). It's non-parametric (no assumptions about distribution).
    
    Args:
        expected: Reference distribution
        actual: Current distribution
    
    Returns:
        Tuple of (ks_statistic, p_value)
        - ks_statistic: Max CDF distance (0 to 1)
        - p_value: Probability distributions are same (<0.05 → reject null)
        
    TODO #12: Implement KS test (5 minutes)
    Hint: Use scipy.stats.ks_2samp(expected, actual)
    """
    # TODO: Perform KS test
    ks_stat, p_value = None, None  # stats.ks_2samp(expected, actual)
    
    logger.info(f"KS test: statistic={ks_stat:.4f}, p-value={p_value:.4f}")
    
    return ks_stat, p_value


# ═══════════════════════════════════════════════════════════════════════════
# EXERCISES (After implementing all TODOs above)
# ═══════════════════════════════════════════════════════════════════════════

"""
Once you've completed all TODOs, test your implementations:

1. Load California Housing data
2. Introduce synthetic outliers and missing values (see ch00_data_prep notebook)
3. Run outlier detection (IQR vs Z-score — compare counts)
4. Analyze missing patterns (check if MAR or MCAR)
5. Compare imputation strategies (which gives lowest MAE?)
6. Test SMOTE on a classification version of the problem
7. Simulate production drift: shift MedInc by +20% and compute PSI

Expected Results:
- IQR should detect more outliers than Z-score (masking effect)
- KNN imputation should outperform mean/median (~2-3k MAE improvement)
- PSI > 0.2 for +20% MedInc shift (significant drift detected)
"""
