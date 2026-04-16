# Real-exchange-rate-elasticity-of-leverage-for-sovereign-balance-sheet

# README — `reer_leverage_elasticity.py`

## Purpose

`reer_leverage_elasticity.py` computes the **accounting exchange-rate elasticity of leverage** from a balance-sheet snapshot, with **no embedded data-retrieval assumptions**.

The module is built around the balance-sheet approach used in IMF work on currency and maturity mismatches, where vulnerability is assessed from **stocks of assets and liabilities** and their sensitivity to exchange-rate shocks, not from regression estimates alone. The broader conceptual motivation also follows Goldstein and Turner: liability-side foreign-currency exposure is incomplete unless asset-side hedges and future income/expenditure hedges are considered.

In symbols, for leverage

\[
\lambda(q)=\frac{L(q)}{A(q)},
\]

where `q` is the chosen exchange-rate driver,

\[
\varepsilon^{acct}_{\lambda,q}
= \frac{\partial \ln \lambda}{\partial \ln q}
= \frac{\partial \ln L}{\partial \ln q}-\frac{\partial \ln A}{\partial \ln q}.
\]

This is a **local accounting sensitivity** implied by the balance sheet at a point in time. It is **not** a regression coefficient. IMF balance-sheet work explicitly motivates this stock-based approach and emphasizes sectoral balance sheets, currency mismatches, and off-balance-sheet exposures. Goldstein–Turner’s mismatch concept also depends on the sensitivity of assets and liabilities, and more broadly future income and expenditure flows, to exchange-rate changes. citeturn786757view0turn548280view0turn548280view1

---

## Design summary

The module has **two layers**, both computing the same economic object.

### 1. Generic snapshot engine

This layer works from **current domestic-currency values** and **component-level elasticities**.

If a balance-sheet side is split into components `X_i`, each with current value `X_i` and local elasticity `eta_i`, then

\[
\frac{\partial \ln X}{\partial \ln q}
= \sum_i \left(\frac{X_i}{X}\right) \eta_i.
\]

So leverage elasticity is computed as

\[
\varepsilon^{acct}_{\lambda,q}
= \sum_{i \in L} w_i^L \eta_i^L - \sum_{j \in A} w_j^A \eta_j^A.
\]

Use this layer for realistic balance sheets with:
- pure domestic items (`eta = 0`)
- pure FX items (`eta = 1`)
- partial indexation or imperfect pass-through (`0 < eta < 1`)
- rare negative sensitivities (`eta < 0`) when economically justified

### 2. Exact linear benchmark model

This layer implements the canonical textbook case:

\[
L(q)=L_D+qL_F^*, \qquad A(q)=A_D+qA_F^*.
\]

Then the exact local leverage elasticity is

\[
\varepsilon^{acct}_{\lambda,q}
= \frac{qL_F^*}{L(q)}-\frac{qA_F^*}{A(q)}
= s_L^{FX}(q)-s_A^{FX}(q).
\]

Use this layer when you want:
- the clean benchmark model
- exact finite-move calculations
- a closed-form check for the generic engine

The two layers are therefore **not different theories**. The second is the exact special case nested inside the first.

---

## What the module contains

### `BalanceSheetValidationError`
Custom exception raised when inputs are invalid or internally inconsistent.

### Internal validators
- `_require_finite_non_negative(value, name)`
- `_require_finite(value, name)`
- `_require_positive(value, name)`

These enforce the numerical conditions required by the formulas:
- component values cannot be negative
- exchange-rate levels must be strictly positive
- totals for assets and liabilities must be strictly positive where division or logs are used

### `ExposureComponent`
Immutable dataclass representing one current asset or liability component.

Fields:
- `name: str`
- `current_value: float`
- `real_fx_elasticity: float`
- `metadata: Mapping[str, object]`

Interpretation:
- `current_value` is the **current domestic-currency value** of the component
- `real_fx_elasticity` is the local elasticity of that value with respect to `q`

Examples:
- domestic fixed-currency debt: `0.0`
- pure FX debt revaluing one-for-one with `q`: `1.0`
- partially indexed liability: e.g. `0.6`

Validation in `__post_init__` ensures:
- `name` is a non-empty string
- `current_value` is finite and non-negative
- `real_fx_elasticity` is finite

### `SideBreakdown`
Immutable dataclass holding a tuple of `ExposureComponent` objects for either assets or liabilities.

Key methods and properties:
- `from_iterable(...)` — constructs a side from component objects
- `from_records(...)` — constructs a side from dict-like records
- `total_value` — sum of component values; must be strictly positive
- `weights()` — current value weights by component name
- `side_real_fx_elasticity()` — value-weighted average elasticity of the side
- `contribution_table()` — component-level decomposition of weights and contributions

Validation in `__post_init__` ensures:
- the side is not empty
- component names are unique

### `BalanceSheetSnapshot`
Immutable dataclass representing the current balance sheet.

Fields:
- `assets: SideBreakdown`
- `liabilities: SideBreakdown`
- `label: str | None`

Computed properties:
- `total_assets`
- `total_liabilities`
- `leverage`
- `asset_real_fx_elasticity`
- `liability_real_fx_elasticity`
- `leverage_real_fx_elasticity`

Method:
- `summary()` returns a compact dict of the key outputs
- `approximate_leverage_change_for_reer_move(delta_log_q)` returns the first-order approximation
  
  \[
  d\ln \lambda \approx \varepsilon^{acct}_{\lambda,q} \cdot d\ln q.
  \]

This object is the core “production” representation for current balance-sheet analysis.

### `LinearOneForOneRevaluationBalanceSheet`
Immutable dataclass implementing the exact linear benchmark.

Inputs:
- `liabilities_domestic_fixed`
- `liabilities_fx_notional`
- `assets_domestic_fixed`
- `assets_fx_notional`

Methods:
- `liabilities(q)`
- `assets(q)`
- `leverage(q)`
- `liability_fx_share(q)`
- `asset_fx_share(q)`
- `leverage_real_fx_elasticity(q)`
- `dlog_leverage(q0, q1)`
- `dlog_q(q0, q1)`
- `finite_move_elasticity(q0, q1)`
- `scenario(q)`

This class is useful when the exact balance-sheet law of motion is known and one wants exact finite-move calculations rather than only local approximations.

### `build_standard_snapshot_from_current_values(...)`
Convenience constructor for the simplest current snapshot with four components:
- liabilities domestic fixed
- liabilities FX sensitive
- assets domestic fixed
- assets FX sensitive

This is useful when the user already has current domestic-currency values and does not need to work with foreign-currency notionals.

---

## How each formula appears in code

### Generic snapshot formula
In `SideBreakdown.side_real_fx_elasticity()`:

```python
return sum((c.current_value / total) * c.real_fx_elasticity for c in self.components)
```

This is exactly the value-weighted average elasticity formula for one balance-sheet side.

In `BalanceSheetSnapshot.leverage_real_fx_elasticity`:

```python
return self.liability_real_fx_elasticity - self.asset_real_fx_elasticity
```

This implements

\[
\varepsilon^{acct}_{\lambda,q}=\varepsilon^{acct}_{L,q}-\varepsilon^{acct}_{A,q}.
\]

### Exact linear benchmark
In `LinearOneForOneRevaluationBalanceSheet`, liabilities and assets are defined as

```python
L(q) = liabilities_domestic_fixed + q * liabilities_fx_notional
A(q) = assets_domestic_fixed + q * assets_fx_notional
```

The FX-sensitive shares are then

```python
(q * liabilities_fx_notional) / L(q)
(q * assets_fx_notional) / A(q)
```

and leverage elasticity is implemented as their difference.

---

## Minimal usage examples

### Generic snapshot

```python
from reer_leverage_elasticity import ExposureComponent, SideBreakdown, BalanceSheetSnapshot

assets = SideBreakdown.from_iterable([
    ExposureComponent("domestic_assets", 100.0, 0.0),
    ExposureComponent("export_receivables_partial", 50.0, 0.6),
])

liabilities = SideBreakdown.from_iterable([
    ExposureComponent("local_debt", 70.0, 0.0),
    ExposureComponent("fx_debt", 30.0, 1.0),
    ExposureComponent("fx_linked_local_bond", 20.0, 0.8),
])

snap = BalanceSheetSnapshot(assets=assets, liabilities=liabilities)
print(snap.summary())
```

### Exact benchmark model

```python
from reer_leverage_elasticity import LinearOneForOneRevaluationBalanceSheet

model = LinearOneForOneRevaluationBalanceSheet(
    liabilities_domestic_fixed=60.0,
    liabilities_fx_notional=20.0,
    assets_domestic_fixed=120.0,
    assets_fx_notional=30.0,
)

print(model.scenario(1.5))
```

---

## Tests

`test_reer_leverage_elasticity.py` checks three things:

1. **Consistency of the two layers**  
   The exact linear model and the generic snapshot produce the same leverage and local elasticity when evaluated at the same current point.

2. **Local derivative accuracy**  
   The exact local elasticity matches a very small finite-difference calculation.

3. **Support for partial indexation**  
   The generic engine correctly handles components with elasticities between 0 and 1.

These tests are the right minimum control set because they verify both the theory-to-code mapping and the internal numerical consistency of the implementation.

---

## Important implementation choices

- **No data-retrieval assumptions**: the module only consumes validated in-memory inputs.
- **`q` is abstract by design**: it can be a real exchange rate, bilateral real rate, or nominal exchange rate, but the valuation convention must match the chosen `q`.
- **Local vs finite-move distinction**: the snapshot engine computes local accounting elasticities; exact finite-move analytics are provided only for the linear benchmark model.
- **No hidden econometrics**: this module computes accounting sensitivities implied by balance-sheet structure. It does not estimate empirical pass-through from historical data.

---

## References

1. IMF, *A Balance Sheet Approach to Financial Crisis* (2002): emphasizes stock vulnerabilities from maturity, currency, and capital-structure mismatches and the role of sectoral balance sheets.  
   https://www.imf.org/en/publications/wp/issues/2016/12/30/a-balance-sheet-approach-to-financial-crisis-16167

2. IMF, *The Balance Sheet Approach and its Applications at the Fund* (2003): explains why sectoral balance sheets, off-balance-sheet positions, reserve adequacy, and foreign-currency exposures matter for surveillance and crisis analysis.  
   https://www.imf.org/external/np/pdr/bal/2003/eng/063003.htm

3. Goldstein and Turner, *Measuring Currency Mismatch and Aggregate Effective Currency Mismatch* (2017 bibliographic abstract): states that currency mismatch depends on the sensitivity of financial assets and liabilities, and also future income and expenditure flows, to exchange-rate changes.  
   https://ideas.repec.org/h/wsi/wschap/9789814749589_0012.html
