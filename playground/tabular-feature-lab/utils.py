"""
Shared utilities for feature-engineering notebooks.

Import from any notebook in this directory:
    from utils import load_fuel_economy_data, debug_transformer, TextPipeline, ...
"""

import builtins

import numpy as np
import pandas as pd

# ── Data Loading ──────────────────────────────────────────────────────────────

def to_tz(df_, time_col, tz_offset, tz_name):
    """Convert naive date strings to timezone-aware datetimes, grouped by offset.

    📊 TIMEZONE CONVERSION FLOW
    [ Raw String ]          [ Extract Offset ]       [ to_tz() ]
    2013-01-01 00:00 EDT -> Offset: 'EST5EDT'  -> Localize & Convert to 'America/New_York'
    """
    return (
        df_
        .groupby(tz_offset)[time_col]
        .transform(lambda s: pd.to_datetime(s)
                              .dt.tz_localize(s.name, ambiguous=True)
                              .dt.tz_convert(tz_name))
    )


def load_fuel_economy_data() -> pd.DataFrame:
    """Download EPA fuel economy data and parse timezone info on createdOn."""
    url = "https://www.fueleconomy.gov/feg/epadata/vehicles.csv"
    cols = [
        "year", "make", "model", "trany", "drive", "VClass", "eng_dscr",
        "barrels08", "city08", "comb08", "range", "evMotor", "cylinders",
        "displ", "fuelCost08", "fuelType", "highway08", "trans_dscr", "createdOn",
    ]
    raw = pd.read_csv(url, low_memory=False)
    return (
        raw
        .loc[:, cols]
        .assign(
            offset=(
                raw.createdOn.str.extract(r"\d\d:\d\d (?P<offset>[A-Z]{3}?)")
                ["offset"].replace("EDT", "EST5EDT")
            ),
            str_date=(
                raw.createdOn.str.slice(4, 19) + " " + raw.createdOn.str.slice(-4)
            ),
            createdOn=lambda df_: to_tz(df_, "str_date", "offset", "America/New_York"),
        )
        .drop(columns=["offset", "str_date"])
    )


# ── Column Groups ─────────────────────────────────────────────────────────────

# Columns used for feature engineering across all EPA notebooks
CAT_COLS = [
    "make", "model", "trany", "drive",
    "VClass", "eng_dscr", "evMotor", "fuelType", "trans_dscr",
]
LOW_CARDINALITY_COLS  = ["VClass", "drive", "fuelType", "trany"]
HIGH_CARDINALITY_COLS = ["make", "model", "eng_dscr", "evMotor", "trans_dscr"]


# ── Pipeline Helpers ──────────────────────────────────────────────────────────

def debug_transformer(X, name):
    """In-pipeline debug hook.

    Stores the current pipeline output in a builtins variable so it is
    accessible from the notebook after the pipeline runs:
        pipeline.fit(X_train, y_train)
        tmp_X   # <- available in notebook namespace
    """
    setattr(builtins, name, X)
    return X


def combine_str_cols_transformer(X, cols, new_col_name):
    """Concatenate multiple string columns into one space-joined column.

    Used as the first step inside TextPipeline so TF-IDF sees a single string.
    """
    return X.assign(
        **{new_col_name: X[cols].fillna("").agg(" ".join, axis="columns")}
    )[new_col_name]


# ── Custom Sklearn Transformers ───────────────────────────────────────────────

from sklearn.base import BaseEstimator, TransformerMixin  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer  # noqa: E402


class TextPipeline(BaseEstimator, TransformerMixin):
    """Multi-column TF-IDF transformer with PCA dimensionality reduction.

    📊 FLOW:
    [multiple string cols] → concatenate → TF-IDF → dense array → PCA(n_components)

    Wraps everything as a single sklearn-compatible step so it can sit inside
    a ColumnTransformer alongside numeric and categorical transformers.
    """

    def __init__(self, cat_cols, n_components=10):
        self.cat_cols = cat_cols
        self.n_components = n_components
        self.text_pipeline = Pipeline([
            ("combine_str", FunctionTransformer(
                combine_str_cols_transformer,
                kw_args={"cols": cat_cols, "new_col_name": "all_str"},
            )),
            ("tfidf", TfidfVectorizer()),
            ("make_dense", FunctionTransformer(lambda X: X.toarray())),
            ("pca", PCA(n_components=n_components)),
        ])

    def fit(self, X, y=None):
        self.text_pipeline.fit(X, y)
        return self

    def transform(self, X):
        res = self.text_pipeline.transform(X)
        return pd.DataFrame(res, index=X.index)


class SpacyVectorizer(BaseEstimator, TransformerMixin):
    """Dense spaCy embedding transformer.

    Collapses multiple string columns into one string per row, then maps each
    row through spaCy's average-vector representation (en_core_web_sm, 96-dim).

    Usage:
        from utils import SpacyVectorizer
        vectorizer = SpacyVectorizer(columns=['eng_dscr', 'trans_dscr'])
    """

    def __init__(self, columns):
        import spacy
        self.columns = columns
        self.nlp = spacy.load("en_core_web_sm")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        text = X[self.columns].fillna("").apply(lambda row: " ".join(row), axis="columns")
        return pd.DataFrame([self.nlp(t).vector for t in text], index=X.index)


# ── Preprocessor Factory ──────────────────────────────────────────────────────

from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, TargetEncoder  # noqa: E402


def make_preprocessor(
    low_cardinality_cols=None,
    high_cardinality_cols=None,
    text_transformer=None,
    cat_cols=None,
):
    """Build the standard ColumnTransformer used across notebooks.

    Args:
        low_cardinality_cols:  columns for One-Hot Encoding  (default: LOW_CARDINALITY_COLS)
        high_cardinality_cols: columns for Target Encoding   (default: HIGH_CARDINALITY_COLS)
        text_transformer:      optional TextPipeline or SpacyVectorizer instance
        cat_cols:              columns passed to text_transformer (required when text_transformer is set)

    Returns:
        sklearn ColumnTransformer ready for .fit() / .transform()
    """
    if low_cardinality_cols is None:
        low_cardinality_cols = LOW_CARDINALITY_COLS
    if high_cardinality_cols is None:
        high_cardinality_cols = HIGH_CARDINALITY_COLS

    transformers = [
        ("cyl_imputer",    SimpleImputer(strategy="constant", fill_value=0),     ["cylinders"]),
        ("displ_imputer",  SimpleImputer(strategy="median"),                      ["displ"]),
        ("one_hot_encoder",
         OneHotEncoder(drop="first", max_categories=10, sparse_output=False, handle_unknown="ignore"),
         low_cardinality_cols),
        ("target_encoder",
         TargetEncoder(target_type="continuous", random_state=42),
         high_cardinality_cols),
    ]
    if text_transformer is not None and cat_cols is not None:
        transformers.append(("text", text_transformer, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="passthrough")
