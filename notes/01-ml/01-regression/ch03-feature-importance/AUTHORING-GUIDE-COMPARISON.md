# Feature Engineering Restructure — Authoring Guide Analysis & Plan

**Date:** April 29, 2026  
**Purpose:** Document why Ch.3 needs restructuring and provide implementation roadmap  
**Status:** Analysis Complete → Ready for Implementation

---

## Executive Summary (50 words)

Ch.3 Feature Engineering needs restructuring from concept-stacked (§Pearson → §Scaling → §VIF) to workflow-based (§Inspect → §Audit → §Transform → §Validate). ML authoring guide needs addendum for procedural chapters. Implementation requires 10-15 hours across 4 phases with minimal LLM calls.

---

## Key Finding: The Authoring Guide Needs an Addendum for Procedural Chapters

The current authoring guide is **concept-focused** (tool → failure → fix → next tool).  
The feature engineering restructure plan is **workflow-focused** (observe → diagnose → decide → implement).

**Both patterns are valid.** The issue is the authoring guide doesn't explicitly support procedural chapters where the practitioner follows a decision tree.

---

## What the Authoring Guide DOES Cover (Already Strong)

### ✅ Pedagogical Patterns Already Present

| Pattern | Status in Guide | Examples |
|---------|-----------------|----------|
| **Failure-first pedagogy** | ✅ Explicit rule | "Tool → specific failure → minimal fix → that fix's failure → next tool" |
| **Numerical walkthroughs** | ✅ Canonical example | "3–5 row subset, step-by-step arithmetic, verification sentence" |
| **Forward/backward linking** | ✅ Mandatory | "Every concept links to where introduced and where it reappears" |
| **Progress check sections** | ✅ Template provided | "✅ Unlocked / ❌ Still can't solve" |
| **Diagnostic flowcharts** | ✅ Required | "Mermaid diagnostic flowchart in What Can Go Wrong section" |
| **Code snippets** | ✅ Covered | "Minimal but complete" + notebook mirroring |
| **Real data requirement** | ✅ Explicit rule | "California Housing real numbers, not toy datasets" |
| **Callout system** | ✅ Defined | 💡⚠️⚡📖➡️ with specific use cases |

### ✅ Voice & Style Rules (Match the Restructure Plan)

- **Second-person practitioner voice** ✅
- **Direct, time-efficient tone** ✅
- **No academic hedging** ✅
- **Every formula gets verbal gloss** ✅
- **Walkthroughs with actual numbers** ✅

---

## What the Authoring Guide DOES NOT Cover (Needs Addition)

### ❌ Workflow-Based Chapter Structure

The current template assumes chapters organized by **concepts**:
```
§1 Core Idea
§2 Running Example
§3 The Math
§4 Step by Step
§5 Hyperparameter Dial
§6 What Can Go Wrong
```

The restructure plan organizes by **workflow phases**:
```
§1 The Workflow (Overview)
§2 Phase 1: Inspect Features → DECISION checkpoint
§3 Phase 2: Check Multicollinearity → DECISION checkpoint
§4 Phase 3: Apply Transformations
§5 Phase 4: Validate
```

**Gap:** No guidance for when/how to use phase-based structure.

---

### ❌ Decision Checkpoint Pattern

**What the plan adds:**
```markdown
### 2.5 DECISION CHECKPOINT

**What you just saw:**
- Feature has skewness = 2.4
- IQR/std = 3.2 (heavy outliers)

**What it means:**
- Right-skewed distribution
- Long tail will compress most data near zero after StandardScaler

**What to do next:**
→ Apply log1p before StandardScaler
→ Verify transformed distribution is more symmetric
```

**Current guide equivalent:** Closest is "What Can Go Wrong" diagnostic flowcharts, but those come at END of chapter, not after each section.

**Gap:** No explicit pattern for mid-section decision checkpoints.

---

### ❌ Code Snippet Placement Strategy

**Current guide:** "Code blocks are minimal but complete" + "Notebook mirrors README"

**What the plan adds:**
- **Code snippets at END of each concept section** (not just in notebook)
- **Example usage blocks** showing "here's how you actually call this"
- **Decision logic in code comments** (`if skew > 1.0: print("→ Use log1p")`)

**Gap:** No explicit rule about when to inline code in README vs defer to notebook.

---

### ❌ Workflow vs Concept Chapter Distinction

**Current guide treats all chapters the same.** The template applies uniformly.

**Reality:** Some chapters are conceptual (linear regression), others are procedural (feature engineering, data validation, hyperparameter tuning).

**Gap:** No guidance on:
- When to use workflow-based structure vs concept-based
- How to identify if a chapter is procedural
- What changes in the template for procedural chapters

---

## Comparison Table: Current Guide vs Restructure Plan

| Dimension | Current Authoring Guide | Feature Engineering Plan | Match? |
|-----------|------------------------|--------------------------|--------|
| **Primary organizing principle** | Concept progression | Workflow phases | ❌ Different |
| **Section structure** | Core Idea → Math → Examples | Inspect → Audit → Transform → Validate | ❌ Different |
| **Decision points** | End of chapter (diagnostic flowchart) | After each phase (checkpoint) | ⚠️ Partial overlap |
| **Code placement** | In notebook, referenced from README | Code snippets inline in README | ⚠️ Different emphasis |
| **Numerical walkthroughs** | Mandatory, 3-5 rows | Mandatory, with decisions annotated | ✅ Same principle |
| **Real data requirement** | California Housing only | California Housing with visualizations | ✅ Same |
| **Failure-first pedagogy** | Explicit rule | Implicit (show what breaks → decide) | ✅ Same spirit |
| **Progress tracking** | § Progress Check at end | Phase-level progress + final check | ⚠️ More granular |
| **Visualization strategy** | Demonstrate what prose can't | Show distributions that inform decisions | ✅ Same principle |

**Legend:**
- ✅ = Plans align
- ⚠️ = Same spirit, different execution
- ❌ = Fundamental difference requiring guidance update

---

## Recommendation: Add "Workflow-Based Chapter Pattern" Section

### Proposed Addition to Authoring Guide

Add new section after "Chapter README Template":

```markdown
## Workflow-Based Chapter Pattern (Procedural Chapters)

> **When to use:** Chapters covering procedures practitioners follow (feature engineering, 
> data validation, model diagnostics, hyperparameter tuning) should use workflow-based structure 
> instead of concept-based structure.

### Identifying Procedural Chapters

A chapter is workflow-based if:
- ✅ It teaches a **sequence of decisions** more than a single concept
- ✅ Practitioner asks "what should I do next?" after each section
- ✅ Multiple tools/techniques are chosen based on data characteristics
- ✅ The chapter reads like a troubleshooting guide, not a concept introduction

**Examples:**
- **Workflow-based:** Feature Engineering (inspect → decide scaler → check VIF → transform)
- **Concept-based:** Linear Regression (concept → math → training → evaluation)

### Modified Template for Workflow Chapters

```
# Ch.N — [Topic Name]

[Same header: story, curriculum context, notation]

---

## 0 · The Challenge — Where We Are
[Same as concept-based template]

## 1 · The Workflow at a Glance
[Numbered list or flowchart showing all phases]

## 2 · Phase 1: [Action Verb] (e.g., "Inspect Features")

### 2.1 What to Look For
[Diagnostic criteria with thresholds]

### 2.2 How to Measure It
[Code snippet showing inspection loop]

### 2.3 Visual Diagnosis
[Histograms, heatmaps, plots with real data]

### 2.4 DECISION CHECKPOINT

**What you just saw:** [Observation from data]
**What it means:** [Interpretation]
**What to do next:** [Action with specific choice]

[Repeat for all phases]

## N-1 · The Complete Decision Tree
[Mermaid flowchart showing all phases + decisions integrated]

## N · Progress Check — What We Can Solve Now
[Same as concept-based template]
```

### Key Differences from Concept-Based Template

| Element | Concept-Based | Workflow-Based |
|---------|---------------|----------------|
| **§1 content** | Core Idea (2-3 sentences) | The Workflow (numbered phases) |
| **Section headers** | Nouns (The Math, The Hyperparameter Dial) | Action verbs (Inspect, Audit, Transform) |
| **Decision points** | End (What Can Go Wrong) | After each phase (Decision Checkpoint) |
| **Code placement** | Primarily in notebook | Inline snippets + notebook |
| **Progress tracking** | Final section only | Phase-level + final |

### Code Snippet Guidelines for Workflow Chapters

**Rule 1: Each phase ends with executable code showing that phase's workflow**

```python
# ✅ Good: Phase 1 code snippet (inspection loop)
for col in numeric_cols:
    skew = df[col].skew()
    iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
    
    # Plot side-by-side: raw vs log-transformed
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df[col], bins=50, edgecolor='black')
    axes[1].hist(np.log1p(df[col]), bins=50, edgecolor='black', color='green')
    
    # DECISION LOGIC
    if abs(skew) > 1.0:
        print(f"{col}: Skew={skew:.2f} → log1p + StandardScaler")
    elif iqr / df[col].std() > 2.5:
        print(f"{col}: Heavy outliers → RobustScaler")
    else:
        print(f"{col}: Symmetric → StandardScaler")
```

**Rule 2: Decision logic appears in code comments, not just prose**

```python
# ✅ Good: Inline decision annotation
if vif > 10:
    print(f"{feat}: VIF={vif:.1f} ❌ SEVERE - Drop or combine")
elif vif > 5 and target_corr < 0.3:
    print(f"{feat}: VIF={vif:.1f}, weak signal → Drop candidate")
else:
    print(f"{feat}: VIF={vif:.1f} ✅ OK")
```

**Rule 3: Example usage blocks after defining functions**

```python
# Define utility
def inspect_feature_distribution(df, col):
    # ... implementation ...

# Example usage — show it in action
inspect_feature_distribution(housing_df, 'AveRooms')
# Output: "AveRooms: Skew=2.4 → log1p + StandardScaler"
```

### Decision Checkpoint Format

Every checkpoint follows this 3-part structure:

```markdown
### N.M DECISION CHECKPOINT

**What you just saw:**
- [Observation 1 with specific numbers]
- [Observation 2 with specific numbers]

**What it means:**
- [Interpretation of observations]
- [Why this matters for the model]

**What to do next:**
→ [Action 1: specific, executable step]
→ [Action 2: validation to perform]
```

**Example from feature engineering:**

```markdown
### 2.4 DECISION CHECKPOINT

**What you just saw:**
- `AveRooms` has skewness = 2.4 (right-skewed)
- IQR/std = 3.2 (heavy outliers)
- Log-transformed histogram is more symmetric

**What it means:**
- StandardScaler on raw data would compress 95% of values near zero
- The few extreme outliers (20+ rooms) would dominate gradient updates
- Log transform decompresses the main mass, shrinks outliers

**What to do next:**
→ Apply `np.log1p(df['AveRooms'])` before StandardScaler
→ Verify: transformed skewness should be < 0.5
→ Add to transformation pipeline (Phase 3)
```

### Visualization Strategy for Workflow Chapters

**Principle:** Show the data characteristics that inform each decision.

**Required plots:**
1. **Inspection phase:** Histograms (raw vs transformed) with annotations
2. **Multicollinearity phase:** Correlation heatmap + VIF bar chart
3. **Transformation phase:** Before/after comparison plots
4. **Validation phase:** Importance rankings + decision convergence

**Format rules:**
- Every plot shows **real California Housing data** (or track-specific dataset)
- Annotate with **decision thresholds** (vertical lines, horizontal bars)
- Include **decision text** directly on plot: "Skew=2.4 → Use log1p"
- Use consistent color scheme: green (OK), orange (caution), red (action required)

### When NOT to Use Workflow Structure

**Stick with concept-based structure for:**
- Chapters introducing single algorithms (CNNs, Transformers, SVMs)
- Chapters where the "workflow" is just "train → evaluate" (no branching decisions)
- Mathematical foundations (MLE, backprop derivation)
- Chapters with single hyperparameter dial (learning rate tuning)

**Use workflow structure for:**
- Multi-tool selection processes (feature selection, scaler choice, regularization)
- Diagnostic procedures (data quality audit, model debugging)
- Tuning strategies with decision trees (hyperparameter search paths)
- Chapters answering "what should I check first?" questions

---

## Impact Assessment: Should Existing Chapters Be Restructured?

| Chapter | Current Structure | Should Restructure? | Reason |
|---------|------------------|---------------------|--------|
| Ch.3 Feature Importance | Concept-based | ✅ **YES** | Procedural workflow (inspect → VIF → transform → validate) |
| Ch.0 Data Prep | Concept-based | ✅ **YES** | Procedural workflow (missing → outliers → scaling) |
| Ch.1 Linear Regression | Concept-based | ❌ **NO** | Single algorithm, not procedural |
| Ch.2 Multiple Regression | Concept-based | ❌ **NO** | Extension of Ch.1, not procedural |
| Ch.7 Hyperparameter Tuning | Concept-based | ⚠️ **CONSIDER** | Has decision tree (LR → batch → optimizer → ...) |
| Ch.8 Data Validation | Concept-based | ✅ **YES** | Procedural workflow (schema → distribution → drift) |

**Recommendation:** Restructure 2-3 high-impact procedural chapters as proof-of-concept before updating authoring guide.

---

## Proposed Action Plan

### Phase 1: Pilot Implementation (Ch.3 Feature Engineering)
- [ ] Restructure Ch.3 following workflow-based plan
- [ ] Generate all required visualizations (histograms, heatmaps, VIF charts)
- [ ] Add decision checkpoints after each phase
- [ ] Inline code snippets showing each phase's workflow
- [ ] Test: Can a reader follow top-to-bottom without jumping around?

### Phase 2: Extract Patterns
- [ ] Document what worked / what didn't from Ch.3 restructure
- [ ] Refine decision checkpoint format
- [ ] Establish visualization standards for workflow chapters
- [ ] Create reusable code snippet templates

### Phase 3: Update Authoring Guide
- [ ] Add "Workflow-Based Chapter Pattern" section
- [ ] Include decision checkpoint template
- [ ] Add code snippet placement rules
- [ ] Provide identification criteria (when to use which structure)
- [ ] Update grand_solution template to accommodate both patterns

### Phase 4: Extend to Other Procedural Chapters
- [ ] Apply to Ch.0 Data Prep
- [ ] Apply to Ch.8 Data Validation (if exists)
- [ ] Apply to Ch.7 Hyperparameter Tuning (evaluate if needed)

---

## Success Metrics

The authoring guide update succeeds if:

1. ✅ **Discoverability:** Authors can identify whether their chapter should be workflow-based or concept-based
2. ✅ **Consistency:** All procedural chapters follow same decision checkpoint pattern
3. ✅ **Usability:** Readers can execute workflows without external references
4. ✅ **Compatibility:** Workflow chapters still satisfy all existing style rules (failure-first, numerical walkthroughs, forward/backward links)
5. ✅ **Maintainability:** Grand solutions can synthesize both workflow and concept chapters into coherent narrative

---

## Conclusion

**The authoring guide does NOT need wholesale replacement.**

It needs a **targeted addendum** for procedural chapters that:
- Defines when to use workflow-based structure
- Provides modified template with decision checkpoints
- Adds code snippet placement rules
- Maintains all existing pedagogical patterns (failure-first, numerical walkthroughs, etc.)

**Next step:** Implement Ch.3 feature engineering restructure as proof-of-concept, then extract patterns into formal authoring guide update.

---

## Industry Standard Tools Integration

**Principle (NEW):** Show manual implementation first (build intuition), then show industry-standard one-liner.

### Pattern to Add to Authoring Guide

After every major technique, add "Industry Standard" callout box:

```markdown
> 💡 **Industry Standard:** `sklearn.preprocessing.StandardScaler`
> 
> ```python
> from sklearn.preprocessing import StandardScaler
> scaler = StandardScaler()
> X_scaled = scaler.fit_transform(X_train)  # One line!
> ```
> 
> **When to use:** Always in production. Manual implementation shown for learning only.
> **Common alternatives:** `RobustScaler`, `MinMaxScaler`, `MaxAbsScaler`
```

### Required Tool Comparisons Per Chapter Type

| Chapter Type | Industry Tools to Show |
|--------------|------------------------|
| Feature Engineering | `StandardScaler`, `RobustScaler`, `ColumnTransformer`, `variance_inflation_factor`, `permutation_importance` |
| Data Validation | `pandas_profiling`, `great_expectations`, `pandera` |
| Hyperparameter Tuning | `GridSearchCV`, `RandomizedSearchCV`, `Optuna` |
| Model Training | `sklearn.model_selection`, `cross_val_score`, `Pipeline` |

### Gap in Current Guide

**Current:** Mentions libraries exist but no pattern for WHEN/HOW to show them

**Needed:** Explicit rule: "After explaining concept with from-scratch code, add Industry Standard callout showing library equivalent"

---

## TODO: Notebook Audit Checklist

**Task:** Verify notebooks explain concepts AND provide production-ready tools

### Per-Notebook Checklist

For every notebook (exercise + solution):
- [ ] Shows manual implementation (learning)
- [ ] Shows sklearn/pandas/numpy equivalent (production)
- [ ] Explains WHEN to use each approach
- [ ] Example: Gradient descent manual → then `model.fit()`
- [ ] Example: Feature scaling manual → then `StandardScaler()`
- [ ] Example: VIF calculation manual → then `statsmodels.VIF`
- [ ] Example: Correlation matrix manual → then `df.corr()`

### Industry Tools Coverage Audit

Pattern per technique:
```python
# Manual (learning): 10 lines showing the math
for i in range(n_iterations):
    gradient = compute_gradient(X, y, w, b)
    w = w - alpha * gradient
    # ...

# Industry standard (production): 1 line
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_train, y_train)  # That's it!
```

**Verify:** Every major technique follows this pattern

---

## Implementation Steps (Minimal LLM Calls)

### Phase 1: Update Authoring Guide (Single Edit)

**Context required:** Lines 150-300 of `notes/01-ml/authoring-guide.md`

**Task:** Insert "Workflow-Based Chapter Pattern" section after "Chapter README Template"

**Content:** See full section proposal above (~600 lines)

**Output:** Updated authoring guide with workflow pattern section

**LLM calls:** 1 (single insert operation)

---

### Phase 2: Restructure Ch.3 Feature Engineering (6 Edits)

**Follow plan in:** `notes/01-ml/01_regression/ch03_feature_importance/PLAN.md`

**Phases:**
1. README restructure (section reorganization)
2. Code snippet insertion (4 locations)
3. Decision checkpoint addition (4 locations)
4. Industry tools callouts (4 locations)
5. Notebook restructure (cell reordering)
6. Visualization integration (manual)

**LLM calls:** 6 total

---

### Phase 3: Apply to Other Procedural Chapters

**Target chapters:**
- Ch.0 Data Prep (7-phase EDA workflow)
- Ch.8 Data Validation (schema → distribution → drift)
- Ch.7 Hyperparameter Tuning (evaluate if procedural)

**Per chapter:** Follow same 6-edit pattern as Ch.3

**LLM calls:** ~18 (6 per chapter × 3 chapters)

---

### Phase 4: Notebook Audits (Manual Review)

**Task:** Review all ML track notebooks for industry tools coverage

**Process:**
1. Identify techniques shown from scratch
2. Verify sklearn/pandas equivalent is shown
3. Add "Industry Standard" callouts where missing
4. Test notebook execution top-to-bottom

**LLM calls:** Variable (depends on missing coverage)

---

## Total Implementation Estimate

| Phase | Effort | LLM Calls |
|-------|--------|-----------|
| Phase 1: Authoring guide update | 3-4 hours | 1 |
| Phase 2: Ch.3 restructure | 10-15 hours | 6 |
| Phase 3: Other chapters | 30-45 hours | 18 |
| Phase 4: Notebook audits | 10-20 hours | Variable |
| **TOTAL** | **53-84 hours** | **25+** |

**Recommendation:** Execute phases sequentially, with review after each phase before proceeding.

---

## Files Modified (All Phases)

### Phase 1 (Authoring Guide)
- `notes/01-ml/authoring-guide.md` (insert ~600 lines)

### Phase 2 (Ch.3 Feature Engineering)
- `notes/01-ml/01_regression/ch03_feature_importance/README.md` (restructure)
- `notes/01-ml/01_regression/ch03_feature_importance/notebook_solution.ipynb` (restructure)
- `notes/01-ml/01_regression/ch03_feature_importance/notebook_exercise.ipynb` (restructure)
- `notes/01-ml/01_regression/ch03_feature_importance/img/` (new plots)

### Phase 3 (Other Chapters)
- `notes/01-ml/01_regression/ch00_data_prep/` (all files)
- `notes/01-ml/03_advanced_topics/ch08_data_validation/` (all files)
- `notes/01-ml/02_advanced_regression/ch07_hyperparameter_tuning/` (if restructured)

### Phase 4 (Notebooks)
- All `notebook_*.ipynb` files across ML track (industry tools callouts)

---

## Success Criteria (Complete)

1. ✅ Authoring guide has workflow-based pattern section with decision checkpoint template
2. ✅ Clear identification criteria (when to use workflow vs concept structure)
3. ✅ Ch.3 Feature Engineering follows workflow structure (proof-of-concept)
4. ✅ All procedural chapters follow same pattern (consistency)
5. ✅ Industry standard tools shown alongside manual implementations (every technique)
6. ✅ Notebooks executable top-to-bottom with both manual + library examples
7. ✅ Decision checkpoints use consistent 3-part format (what/means/next)
8. ✅ All visualizations use real California Housing data with decision annotations

---

## Next Actions

1. ✅ **Review this complete analysis and plan**
2. → **Execute Phase 1:** Update authoring guide (single insert, 3-4 hours)
3. → **Execute Phase 2:** Restructure Ch.3 as proof-of-concept (6 edits, 10-15 hours)
4. → **Checkpoint:** Review Phase 2 output, refine patterns if needed
5. → **Execute Phase 3:** Apply to remaining procedural chapters (18 edits, 30-45 hours)
6. → **Execute Phase 4:** Audit notebooks for industry tools coverage (manual review)

**Priority:** Start with Phase 1 + Phase 2, validate approach before scaling to other chapters.

---

## Appendix: Related Documentation

- **Ch.3 Restructure Plan:** `notes/01-ml/01_regression/ch03_feature_importance/RESTRUCTURE_PLAN.md`
- **Ch.3 Implementation Plan:** `notes/01-ml/01_regression/ch03_feature_importance/PLAN.md`
- **ML Track Authoring Guide Update Plan:** `notes/01-ml/AUTHORING_GUIDE_UPDATE_PLAN.md`
- **Current ML Authoring Guide:** `notes/01-ml/authoring-guide.md`
- **Animation Conventions:** `/memories/repo/animation-conventions.md`
