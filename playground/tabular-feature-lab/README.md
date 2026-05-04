# Tabular Feature Lab

**Source:** LinkedIn Learning
**Topic:** Feature Engineering for Machine Learning
**Status:** Exploratory notebooks for learning feature engineering patterns

## Overview

This directory contains notebooks from a LinkedIn Learning course focused on feature engineering techniques for tabular machine learning. These notebooks demonstrate classical ML preprocessing and feature creation strategies.

## Notebooks

### 1. [categorical_encoding.ipynb](categorical_encoding.ipynb)
**Focus:** Encoding categorical variables for ML models

**Key Techniques:**
- One-Hot Encoding (low cardinality)
- Hash Encoding (high cardinality, streaming)
- Target Encoding (high signal features)
- Label/Ordinal Encoding (tree models)
- Binary, Count, Leave-One-Out encoders

**Dataset:** EPA Fuel Economy (vehicles.csv) - `make`, `model`, `VClass`, `fuelType`

**Learning Focus:**
- When to use which encoder (cardinality considerations)
- Target encoding risks (data leakage)
- Memory trade-offs (One-Hot vs Hash)

---

### 2. [feature_engg.ipynb](feature_engg.ipynb)
**Focus:** Feature transformation and creation strategies

**Key Techniques:**
- Scaling/Normalization (MinMaxScaler, StandardScaler)
- Log and Power Transformations (handle skew)
- Polynomial Features (interaction terms)
- Binning/Discretization
- Date/Time Feature Extraction
- Correlation Analysis & Feature Selection

**Dataset:** EPA Fuel Economy - `cylinders`, `displ`, `city08`, `highway08`, `comb08`

**Learning Focus:**
- Prevent data leakage (fit on train, transform on test)
- Polynomial feature explosion (use sparingly)
- Domain-specific features (MPG ratios, efficiency metrics)
- sklearn Pipeline patterns

---

### 3. [feature_extraction.ipynb](feature_extraction.ipynb)
**Focus:** Text feature extraction with TF-IDF

**Key Concepts:**
- TF-IDF (Term Frequency-Inverse Document Frequency)
- Sparse matrix representations
- Vocabulary management (max_features, stop words)
- N-grams for phrase capture

**Dataset:** EPA Fuel Economy - `eng_dscr`, `trans_dscr` (text fields)

**Learning Focus:**
- Classical NLP feature extraction
- Foundation for modern embeddings
- Bridge to RAG systems (TF-IDF as lightweight retrieval)
- Comparison: TF-IDF vs Word2Vec vs Transformers

---

### 4. [feature_evaluation.ipynb](feature_evaluation.ipynb)
**Focus:** Comparing TF-IDF vs dense text embeddings in an end-to-end pipeline

**Key Techniques:**
- TF-IDF + PCA inside a custom `TextPipeline` sklearn transformer
- spaCy `en_core_web_sm` dense embeddings via `SpacyEmbeddingVectorizer`
- `ColumnTransformer` composing numeric, categorical, and text steps
- Side-by-side R² comparison of feature extraction strategies

**Dataset:** EPA Fuel Economy - multi-column text features (`eng_dscr`, `trans_dscr`, `model`, `evMotor`)

**Learning Focus:**
- When TF-IDF outperforms dense embeddings (and vice versa)
- Wrapping spaCy inside sklearn's `TransformerMixin` for pipeline compatibility
- Trade-offs: vocabulary interpretability vs semantic richness vs latency

---

### 5. [temporal_features.ipynb](temporal_features.ipynb)
**Focus:** Temporal feature engineering for time series and datetime data

**Key Techniques:**
- `feature_engine.datetime.DatetimeFeatures` — extract year, month, day, hour, etc.
- Seasonal decomposition (trend, seasonality, residuals) via statsmodels
- Group-wise `SeasonTransformer` (sklearn-compatible) for per-entity decomposition
- Cyclical encoding: sin/cos month pairs so Dec ↔ Jan are geometrically adjacent
- Linear trend index for capturing overall growth
- Time-respecting train/test split (no shuffle)

**Datasets:**
- EPA Fuel Economy — group-wise seasonal decomposition by car make
- Box-Jenkins Airline Passengers (seaborn `flights`) — challenge: predict monthly passenger counts

**Learning Focus:**
- Never shuffle time series; temporal split prevents leakage
- Cyclical encoding vs raw integer months
- Production pattern: `SeasonTransformer` as a named pipeline step

---

## Connection to AI Portfolio

### Relates to notes/01-ml:
- **Regression chapters**: Feature engineering for continuous targets
- **Classification chapters**: Categorical encoding strategies
- **Feature selection**: Variance thresholds, correlation analysis

### Relates to notes/03-ai:
- **Embedding chapters**: TF-IDF as classical embedding method
- **RAG chapters**: TF-IDF for document retrieval (alternative to dense embeddings)
- **Feature engineering → prompt engineering**: Similar concepts (crafting inputs for better model performance)

### Relates to notes/02-advanced-deep-learning:
- **Sequence chapters**: Classical temporal decomposition vs RNN/LSTM approaches
- **Temporal features** are the tabular precursor to sequential model inputs

### Production Patterns:
All notebooks emphasize preventing **data leakage**:
- Fit transformers/encoders on training data only
- Apply fitted transformers to test/validation data
- Save fitted objects for production inference
- Use sklearn Pipeline to enforce correct order
- **For time series**: never shuffle; use temporal train/test split

---

## Dependencies

### Python Packages:
```
pandas
numpy
scikit-learn
category_encoders  # For advanced categorical encoders
matplotlib
seaborn             # Built-in datasets (flights for temporal challenge)
statsmodels         # Seasonal decomposition (temporal_features.ipynb)
feature_engine      # DatetimeFeatures extractor (temporal_features.ipynb)
spacy               # Dense text embeddings (feature_evaluation.ipynb)
```

### Data Source:
- **EPA Fuel Economy API**: `https://www.fueleconomy.gov/feg/epadata/vehicles.csv`
- Public dataset, no authentication required
- Updated annually by US EPA

---

## Common Pitfalls (Documented in Notebooks)

1. **Data Leakage**: Fitting on entire dataset before train/test split
2. **Memory Issues**: One-Hot encoding high-cardinality features, excessive polynomial features
3. **Target Encoding Risks**: Using target variable without cross-validation
4. **Missing Production Artifacts**: Not saving fitted transformers/encoders
5. **Ignoring Feature Drift**: Not monitoring feature distributions in production

---

## Future Improvements

- [ ] Add end-to-end pipeline examples (data → features → model → evaluation)
- [ ] Show before/after model performance metrics for each technique
- [ ] Compare classical (TF-IDF) vs modern (embeddings) text features
- [ ] Add automated feature selection (RFE, SelectKBest)
- [ ] Document computational costs and memory usage
- [ ] Add cross-validation patterns for target encoding
- [ ] Create visualizations comparing encoding/transformation methods

---

## Usage Notes

These notebooks are **exploratory** - meant for learning patterns, not production-ready code.

**For production use**, adapt patterns to include:
- Proper train/test splits
- sklearn Pipeline for reproducibility
- Saved fitted transformers (pickle/joblib)
- Feature drift monitoring
- Version-controlled feature engineering code

**Language is intentionally vague about course details** to avoid licensing complexities while preserving technical learning.

---

## Future Merge Guidance

**For adding new notebooks to this collection:**

### 1. Security Audit
- [ ] Check for hardcoded API keys, secrets, tokens
- [ ] Verify data sources are public or properly anonymized
- [ ] Ensure no PII in sample data
- [ ] Confirm all data loading is self-contained (inline downloads)

### 2. Documentation Requirements
Add comprehensive header cell to each notebook:
- **Overview** - What the notebook teaches
- **Key Techniques** - Bullet list of methods covered
- **Dataset Context** - What data is used and where it comes from
- **Notes for Future Revision** section with:
  - Security audit results
  - Key learnings (what worked, pitfalls)
  - Production patterns demonstrated
  - Related chapters in notes/
  - Improvements needed (checkbox list)
  - Dependencies
  - Common pitfalls to avoid

### 3. README Updates
- Add new notebook to this README's "Notebooks" section
- Include focus, key techniques, dataset, learning focus
- Update "Connection to AI Portfolio" section if applicable
- Add to "Future Improvements" list if needed

### 4. Directory Naming
- Keep directory name semantic and concise (current: `feature-engineering`)
- Avoid acronyms or course-specific codes
- Use kebab-case (lowercase with hyphens)

### 5. Dependencies
- If notebooks have significant dependencies beyond pandas/numpy/sklearn, add `requirements.txt`
- Document in README under "Dependencies" section
- Consider adding setup scripts (setup.ps1, setup.sh) if complex setup needed

### 6. Language & Attribution
- Keep language vague about course source to avoid licensing issues
- Focus on technical content, not course branding
- Make clear these are "exploratory notebooks for learning"

### 7. Commit Message Template
```
Add [topic] notebooks from [vague source] course

- [notebook-name-1.ipynb]: [one-line description]
- [notebook-name-2.ipynb]: [one-line description]
- Added comprehensive documentation headers
- Security audit: No hardcoded secrets
- Self-contained data loading
```

### 8. Integration Decision
**When to integrate into notes/ chapters:**
- Patterns are novel and not already covered
- Demonstrates best practices worth teaching
- Has clear pedagogical value beyond exploration
- Aligns with existing chapter structure

**When to keep as standalone playground:**
- Exploratory learning without clear integration point
- Covers topics already well-documented in notes/
- Simple technique demonstrations
- Course-specific examples that don't generalize

---

**Example merge workflow:**
```powershell
# 1. Checkout branch with new content
git checkout [new-content-branch]

# 2. Audit and document notebooks (add header cells)

# 3. Update this README with new notebook details

# 4. Security scan
grep -r "api_key\|API_KEY\|secret\|SECRET" *.ipynb

# 5. Merge to main
git checkout main
git merge [new-content-branch] --no-edit

# 6. Commit documentation additions
git add playground/feature-engineering/*.ipynb
git add playground/feature-engineering/README.md
git commit -m "Add [topic] notebooks with comprehensive documentation"
```
