from __future__ import annotations

"""Production-grade implementation of real exchange-rate elasticity of leverage.

The module follows the balance-sheet approach used in IMF work on currency
mismatches and the Goldstein-Turner critique that liability-side foreign-
currency shares alone are incomplete because asset-side hedges matter.

Core identity
-------------
For leverage lambda(q) = L(q) / A(q), where q is the real exchange rate,

the accounting elasticity is

    epsilon_{lambda,q} = d log(lambda) / d log(q)
                       = d log(L) / d log(q) - d log(A) / d log(q)

If each component i of liabilities or assets has a current domestic-currency
value X_i and a real-exchange-rate elasticity eta_i, then the aggregate side
elasticity is the value-weighted average elasticities across components:

    d log(X) / d log(q) = sum_i (X_i / X_total) * eta_i

Hence,

    epsilon_{lambda,q} = sum_{i in L} w_i^L eta_i^L - sum_{j in A} w_j^A eta_j^A

This nests the standard two-bucket case where domestic-currency-fixed items
have elasticity 0 and foreign-currency items that revalue one-for-one with q
have elasticity 1. In that special case:

    epsilon_{lambda,q} = s_L^{FX}(q) - s_A^{FX}(q)

The module offers two layers:
1) Generic local accounting elasticity from current balance-sheet values.
2) Exact finite-move analytics for the linear one-for-one revaluation case.

Notes
-----
- The code is agnostic to data retrieval. Inputs are validated in-memory objects.
- Use a real exchange rate q only if the balance-sheet valuation convention is
  consistent with q. If liabilities/assets are marked using a nominal exchange
  rate, then use the matching nominal rate instead of a REER.
- Elasticities are local objects. Finite-move results are exact only in the
  specialized linear model provided below.
"""

from dataclasses import dataclass, field
from math import isfinite, log
from typing import Iterable, Mapping, Sequence


class BalanceSheetValidationError(ValueError):
    """Raised when the balance-sheet inputs are internally inconsistent."""


def _require_finite_non_negative(value: float, name: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise BalanceSheetValidationError(f"{name} must be a real number.") from exc
    if not isfinite(x):
        raise BalanceSheetValidationError(f"{name} must be finite.")
    if x < 0.0:
        raise BalanceSheetValidationError(f"{name} must be non-negative.")
    return x


def _require_finite(value: float, name: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise BalanceSheetValidationError(f"{name} must be a real number.") from exc
    if not isfinite(x):
        raise BalanceSheetValidationError(f"{name} must be finite.")
    return x


def _require_positive(value: float, name: str) -> float:
    x = _require_finite(value, name)
    if x <= 0.0:
        raise BalanceSheetValidationError(f"{name} must be strictly positive.")
    return x


@dataclass(frozen=True, slots=True)
class ExposureComponent:
    """Single asset or liability component at the current balance-sheet date.

    Parameters
    ----------
    name:
        Human-readable identifier.
    current_value:
        Current domestic-currency value of the component.
    real_fx_elasticity:
        Local elasticity of this component's domestic-currency value with
        respect to the chosen real exchange-rate measure q.

        Examples:
        - 0.0 for domestic-currency fixed instruments.
        - 1.0 for foreign-currency items that revalue one-for-one with q.
        - Between 0 and 1 for partial indexation or imperfect pass-through.
        - Negative values are allowed for components that decline in domestic
          value when q rises, though this should be rare and carefully justified.
    metadata:
        Optional free-form mapping for instrument tags (sector, law, maturity,
        rate type, source system id, etc.).
    """

    name: str
    current_value: float
    real_fx_elasticity: float
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise BalanceSheetValidationError("ExposureComponent.name must be a non-empty string.")
        object.__setattr__(self, "current_value", _require_finite_non_negative(self.current_value, f"{self.name}.current_value"))
        object.__setattr__(self, "real_fx_elasticity", _require_finite(self.real_fx_elasticity, f"{self.name}.real_fx_elasticity"))


@dataclass(frozen=True, slots=True)
class SideBreakdown:
    """Current balance-sheet side (assets or liabilities)."""

    components: tuple[ExposureComponent, ...]

    def __post_init__(self) -> None:
        if len(self.components) == 0:
            raise BalanceSheetValidationError("A balance-sheet side must contain at least one component.")
        seen: set[str] = set()
        for c in self.components:
            if c.name in seen:
                raise BalanceSheetValidationError(f"Duplicate component name detected: {c.name!r}")
            seen.add(c.name)

    @classmethod
    def from_iterable(cls, components: Iterable[ExposureComponent]) -> "SideBreakdown":
        return cls(tuple(components))

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, object]],
        *,
        name_key: str = "name",
        value_key: str = "current_value",
        elasticity_key: str = "real_fx_elasticity",
        metadata_keys: Sequence[str] | None = None,
    ) -> "SideBreakdown":
        components: list[ExposureComponent] = []
        for record in records:
            metadata: dict[str, object] = {}
            if metadata_keys is not None:
                metadata = {k: record[k] for k in metadata_keys if k in record}
            else:
                metadata = {
                    k: v
                    for k, v in record.items()
                    if k not in {name_key, value_key, elasticity_key}
                }
            components.append(
                ExposureComponent(
                    name=str(record[name_key]),
                    current_value=float(record[value_key]),
                    real_fx_elasticity=float(record[elasticity_key]),
                    metadata=metadata,
                )
            )
        return cls.from_iterable(components)

    @property
    def total_value(self) -> float:
        total = sum(c.current_value for c in self.components)
        return _require_positive(total, "side total")

    def weights(self) -> dict[str, float]:
        total = self.total_value
        return {c.name: c.current_value / total for c in self.components}

    def side_real_fx_elasticity(self) -> float:
        total = self.total_value
        return sum((c.current_value / total) * c.real_fx_elasticity for c in self.components)

    def contribution_table(self) -> list[dict[str, float | str]]:
        total = self.total_value
        rows: list[dict[str, float | str]] = []
        for c in self.components:
            weight = c.current_value / total
            rows.append(
                {
                    "name": c.name,
                    "current_value": c.current_value,
                    "weight": weight,
                    "real_fx_elasticity": c.real_fx_elasticity,
                    "elasticity_contribution": weight * c.real_fx_elasticity,
                }
            )
        return rows


@dataclass(frozen=True, slots=True)
class BalanceSheetSnapshot:
    """Current balance-sheet snapshot with local accounting elasticities.

    This object computes the exact local accounting elasticity implied by the
    current composition of assets and liabilities. It does not require exchange-
    rate histories or regression estimation.
    """

    assets: SideBreakdown
    liabilities: SideBreakdown
    label: str | None = None

    @property
    def total_assets(self) -> float:
        return self.assets.total_value

    @property
    def total_liabilities(self) -> float:
        return self.liabilities.total_value

    @property
    def leverage(self) -> float:
        return self.total_liabilities / self.total_assets

    @property
    def asset_real_fx_elasticity(self) -> float:
        return self.assets.side_real_fx_elasticity()

    @property
    def liability_real_fx_elasticity(self) -> float:
        return self.liabilities.side_real_fx_elasticity()

    @property
    def leverage_real_fx_elasticity(self) -> float:
        return self.liability_real_fx_elasticity - self.asset_real_fx_elasticity

    def summary(self) -> dict[str, float | str | None]:
        return {
            "label": self.label,
            "total_assets": self.total_assets,
            "total_liabilities": self.total_liabilities,
            "leverage": self.leverage,
            "asset_real_fx_elasticity": self.asset_real_fx_elasticity,
            "liability_real_fx_elasticity": self.liability_real_fx_elasticity,
            "leverage_real_fx_elasticity": self.leverage_real_fx_elasticity,
        }

    def approximate_leverage_change_for_reer_move(self, delta_log_q: float) -> float:
        """First-order approximation of d log(leverage) for a log move in q.

        Returns
        -------
        float
            Approximate proportional change in leverage:
            d log(lambda) ~= epsilon_{lambda,q} * d log(q)
        """
        dlogq = _require_finite(delta_log_q, "delta_log_q")
        return self.leverage_real_fx_elasticity * dlogq


@dataclass(frozen=True, slots=True)
class LinearOneForOneRevaluationBalanceSheet:
    """Exact finite-move model for linear one-for-one revaluation.

    This is the canonical balance-sheet model behind the textbook result that
    leverage elasticity equals the difference between the current foreign-
    currency-sensitive shares of liabilities and assets.

    Definitions
    -----------
    L(q) = L_domestic_fixed + q * L_fx_notional
    A(q) = A_domestic_fixed + q * A_fx_notional

    where q > 0 is the real exchange rate, measured so that a rise in q is a
    real depreciation and the foreign-currency notional values are mapped into
    domestic-currency values via q.

    The exact local elasticity is:

        epsilon_{lambda,q} = (q * L_fx_notional / L(q)) - (q * A_fx_notional / A(q))

    which is the special case s_L^{FX}(q) - s_A^{FX}(q).
    """

    liabilities_domestic_fixed: float
    liabilities_fx_notional: float
    assets_domestic_fixed: float
    assets_fx_notional: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "liabilities_domestic_fixed", _require_finite_non_negative(self.liabilities_domestic_fixed, "liabilities_domestic_fixed"))
        object.__setattr__(self, "liabilities_fx_notional", _require_finite_non_negative(self.liabilities_fx_notional, "liabilities_fx_notional"))
        object.__setattr__(self, "assets_domestic_fixed", _require_finite_non_negative(self.assets_domestic_fixed, "assets_domestic_fixed"))
        object.__setattr__(self, "assets_fx_notional", _require_finite_non_negative(self.assets_fx_notional, "assets_fx_notional"))
        if self.liabilities_domestic_fixed + self.liabilities_fx_notional <= 0.0:
            raise BalanceSheetValidationError("Total liabilities at q=1 must be positive.")
        if self.assets_domestic_fixed + self.assets_fx_notional <= 0.0:
            raise BalanceSheetValidationError("Total assets at q=1 must be positive.")

    def liabilities(self, q: float) -> float:
        qv = _require_positive(q, "q")
        value = self.liabilities_domestic_fixed + qv * self.liabilities_fx_notional
        return _require_positive(value, "L(q)")

    def assets(self, q: float) -> float:
        qv = _require_positive(q, "q")
        value = self.assets_domestic_fixed + qv * self.assets_fx_notional
        return _require_positive(value, "A(q)")

    def leverage(self, q: float) -> float:
        return self.liabilities(q) / self.assets(q)

    def liability_fx_share(self, q: float) -> float:
        qv = _require_positive(q, "q")
        return (qv * self.liabilities_fx_notional) / self.liabilities(qv)

    def asset_fx_share(self, q: float) -> float:
        qv = _require_positive(q, "q")
        return (qv * self.assets_fx_notional) / self.assets(qv)

    def leverage_real_fx_elasticity(self, q: float) -> float:
        qv = _require_positive(q, "q")
        return self.liability_fx_share(qv) - self.asset_fx_share(qv)

    def dlog_leverage(self, q0: float, q1: float) -> float:
        q0v = _require_positive(q0, "q0")
        q1v = _require_positive(q1, "q1")
        return log(self.leverage(q1v) / self.leverage(q0v))

    def dlog_q(self, q0: float, q1: float) -> float:
        q0v = _require_positive(q0, "q0")
        q1v = _require_positive(q1, "q1")
        return log(q1v / q0v)

    def finite_move_elasticity(self, q0: float, q1: float) -> float:
        """Average elasticity over a finite move from q0 to q1."""
        dlogq = self.dlog_q(q0, q1)
        if dlogq == 0.0:
            raise BalanceSheetValidationError("q0 and q1 must differ for finite_move_elasticity.")
        return self.dlog_leverage(q0, q1) / dlogq

    def scenario(self, q: float) -> dict[str, float]:
        qv = _require_positive(q, "q")
        L = self.liabilities(qv)
        A = self.assets(qv)
        return {
            "q": qv,
            "liabilities": L,
            "assets": A,
            "leverage": L / A,
            "liability_fx_share": self.liability_fx_share(qv),
            "asset_fx_share": self.asset_fx_share(qv),
            "leverage_real_fx_elasticity": self.leverage_real_fx_elasticity(qv),
        }


def build_standard_snapshot_from_current_values(
    *,
    current_liabilities_domestic_fixed: float,
    current_liabilities_fx_sensitive: float,
    current_assets_domestic_fixed: float,
    current_assets_fx_sensitive: float,
    label: str | None = None,
) -> BalanceSheetSnapshot:
    """Convenience constructor for the standard domestic-vs-FX split.

    Inputs are current domestic-currency values, not foreign-currency notionals.
    The resulting local accounting elasticity equals the current FX-sensitive
    liability share minus the current FX-sensitive asset share.
    """
    liabilities = SideBreakdown.from_iterable(
        [
            ExposureComponent("liabilities_domestic_fixed", current_liabilities_domestic_fixed, 0.0),
            ExposureComponent("liabilities_fx_sensitive", current_liabilities_fx_sensitive, 1.0),
        ]
    )
    assets = SideBreakdown.from_iterable(
        [
            ExposureComponent("assets_domestic_fixed", current_assets_domestic_fixed, 0.0),
            ExposureComponent("assets_fx_sensitive", current_assets_fx_sensitive, 1.0),
        ]
    )
    return BalanceSheetSnapshot(assets=assets, liabilities=liabilities, label=label)


__all__ = [
    "BalanceSheetValidationError",
    "ExposureComponent",
    "SideBreakdown",
    "BalanceSheetSnapshot",
    "LinearOneForOneRevaluationBalanceSheet",
    "build_standard_snapshot_from_current_values",
]
