"""
sovereign_salm.py
=================
Sovereign Asset and Liability Management (SALM) — Economic Balance Sheet Model
===============================================================================

PURPOSE
-------
Compute the sovereign economic balance sheet, leverage ratio lambda(q) = L(q)/A(q),
and the elasticity of the leverage ratio with respect to the real exchange rate q:

    epsilon_{lambda,q} = d ln(lambda) / d ln(q)

THEORETICAL FOUNDATIONS
------------------------
1. Sovereign Economic Balance Sheet (IMF WP/12/241, 2012; IMF WP/19/290, 2019)
   Assets:      A(q) = A_fin_0 * (q/q0)^delta_A + PV(R_T(q)) + PV(R_N(q))
   Liabilities: L(q) = D_FX_0*(q/q0)^chi + D_LC + PV(G_T(q)) + PV(G_N(q))
   Leverage:    lambda(q) = L(q) / A(q)

2. Revenue Decomposition (IMF GFSM 2014, Chapter 5)
   RT_t = sum_i alpha_i  * R_{i,t}    [tradable-linked revenue]
   RN_t = sum_i (1-alpha_i) * R_{i,t} [nontradable-linked revenue]

   NOTE: alpha_i is the tradable-basis share of revenue instrument i.
   The approved fallback when instrument-level data are unavailable is:
     alpha_i ~ (Exports / GDP)
   This is a proxy, NOT an identity. It must be explicitly flagged.

3. Real Exchange Rate (IMF Finance & Development; BIS QR March 2006)
   q = e * P_star / P   [bilateral RER; higher q = real depreciation]
   FX debt is fundamentally revalued by the nominal exchange rate e.
   Therefore, when the model is expressed in q, the effective elasticity of
   FX debt to q is:
       eta_DFX,q = chi * psi_q_to_e
   where:
       chi        = d ln(D_FX) / d ln(e)       [nominal-FX pass-through]
       psi_q_to_e = d ln(e) / d ln(q)          [mapping from q shock to e shock]
   Under sticky prices over the shock horizon: d ln q = d ln e, so
       psi_q_to_e = 1 and eta_DFX,q = chi.

4. Response Functions (parametric, power-law)
   R_T_t(q) = R_T_t0 * (q/q0)^beta_T_t
   R_N_t(q) = R_N_t0 * (q/q0)^beta_N_t
   G_T_t(q) = G_T_t0 * (q/q0)^gamma_T_t
   G_N_t(q) = G_N_t0 * (q/q0)^gamma_N_t
   D_FX(q)  = D_FX_0 * (q/q0)^(chi * psi_q_to_e)

5. Present Value (IMF TNM/10/02, Escolano 2010)
   PV(X) = sum_{t=1}^T X_t / prod_{s=1}^t (1 + r_s)

6. Leverage Elasticity (analytical via log-differentiation)
   epsilon_{lambda,q} = d ln L / d ln q - d ln A / d ln q
   Each term is a value-weighted average of block elasticities.

SOURCE DISCIPLINE
-----------------
- Source-based definitions are marked [SOURCE].
- Modeling assumptions are marked [ASSUMPTION].
- Implementation decisions are marked [IMPL].

AUTHOR NOTE
-----------
Uses exports/GDP as proxy for alpha_i only when approved by caller and only
when instrument-level tradable-basis shares are unavailable. This is a fallback
approximation, not a theoretical identity.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ============================================================================
# SECTION 1: DATA STRUCTURES
# ============================================================================


@dataclass
class RevenueInstrument:
    """
    [SOURCE] Revenue instrument consistent with IMF GFSM 2014, Chapter 5.
    (https://www.imf.org/external/pubs/ft/gfs/manual/pdf/ch5.pdf)

    Parameters
    ----------
    name : str
        Descriptive label (e.g., "VAT", "corporate_income_tax", "customs_duties").
    amounts : np.ndarray
        Nominal revenue amounts by period t = 1..T in domestic-currency units.
    alpha : float
        [ASSUMPTION] Tradable-basis share in [0, 1].
        Proportion of this revenue instrument whose base moves with the
        tradable sector (and hence with the real exchange rate q).
        If set via the exports/GDP fallback, caller must pass use_proxy=True.
    use_proxy : bool
        If True, alpha was set via exports/GDP proxy. Logged in output.
    """
    name: str
    amounts: np.ndarray          # shape (T,)
    alpha: float                  # in [0, 1]
    use_proxy: bool = False

    def __post_init__(self) -> None:
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(
                f"RevenueInstrument '{self.name}': alpha={self.alpha} "
                "must be in [0, 1]."
            )
        if np.any(~np.isfinite(self.amounts)):
            raise ValueError(
                f"RevenueInstrument '{self.name}': amounts contains "
                "non-finite values."
            )
        if self.use_proxy:
            warnings.warn(
                f"RevenueInstrument '{self.name}': alpha set via exports/GDP "
                "proxy. This is an approximation, not a theoretical identity.",
                UserWarning,
                stacklevel=2,
            )


@dataclass
class ExpenditureInstrument:
    """
    [SOURCE] Government expenditure instrument.

    Parameters
    ----------
    name : str
        Descriptive label (e.g., "wages", "subsidies", "capital_expenditure").
    amounts : np.ndarray
        Nominal expenditure amounts by period t = 1..T.
    phi : float
        [ASSUMPTION] Tradable-basis share of expenditure in [0, 1].
        Captures the fraction of expenditure whose real cost moves with q
        (e.g., imported goods, FX-denominated procurement).
    """
    name: str
    amounts: np.ndarray          # shape (T,)
    phi: float                    # in [0, 1]

    def __post_init__(self) -> None:
        if not (0.0 <= self.phi <= 1.0):
            raise ValueError(
                f"ExpenditureInstrument '{self.name}': phi={self.phi} "
                "must be in [0, 1]."
            )
        if np.any(~np.isfinite(self.amounts)):
            raise ValueError(
                f"ExpenditureInstrument '{self.name}': amounts contains "
                "non-finite values."
            )


@dataclass
class DebtBlock:
    """
    Sovereign debt blocks.

    [ASSUMPTION] Foreign-currency debt (D_FX) is fundamentally revalued by the
    nominal exchange rate e, not directly by the real exchange rate q.
    Therefore, when the model is written in q, FX debt loads on q through the
    chain rule:

        d ln(D_FX) / d ln(q)
        =
        (d ln(D_FX) / d ln(e)) * (d ln(e) / d ln(q))
        =
        chi * psi_q_to_e

    Local-currency debt (D_LC) is invariant to q in the baseline.

    Parameters
    ----------
    D_FX_0 : float
        Foreign-currency debt at baseline q0, expressed in domestic-currency
        equivalent (i.e., already converted at e_0).
    D_LC : float
        Local-currency debt. Invariant to q in baseline.
    chi : float
        [ASSUMPTION] Elasticity of D_FX with respect to the nominal exchange
        rate e. Baseline chi = 1 means FX debt revalues 1-for-1 with e.
        Must be > 0 by convention.
    psi_q_to_e : float
        [ASSUMPTION] Mapping from real-exchange-rate shocks to nominal
        exchange-rate shocks:
            psi_q_to_e = d ln(e) / d ln(q)
        Under sticky prices over the shock horizon, psi_q_to_e = 1 so
        d ln q = d ln e locally.
        Must be > 0 by convention.
    """
    D_FX_0: float
    D_LC: float
    chi: float = 1.0
    psi_q_to_e: float = 1.0

    def __post_init__(self) -> None:
        if self.D_FX_0 < 0:
            raise ValueError("D_FX_0 must be >= 0.")
        if self.D_LC < 0:
            raise ValueError("D_LC must be >= 0.")
        if self.chi <= 0:
            raise ValueError("chi must be > 0.")
        if self.psi_q_to_e <= 0:
            raise ValueError("psi_q_to_e must be > 0.")

    @property
    def eta_DFX_q(self) -> float:
        """
        Effective elasticity of FX debt with respect to the real exchange rate q.

        Returns
        -------
        float
            eta_DFX_q = chi * psi_q_to_e
        """
        return self.chi * self.psi_q_to_e


@dataclass
class FinancialAssets:
    """
    Current public financial assets (reserves, equity stakes, etc.).

    [SOURCE] IMF WP/12/241 includes current financial assets on the sovereign
    economic balance sheet.

    Parameters
    ----------
    A_fin_0 : float
        Value at baseline q0 in domestic-currency units.
    delta_A : float
        [ASSUMPTION] Elasticity of A_fin with respect to q.
        Default 0.0: financial assets are denominated in local currency or
        fully hedged, so they do not revalue with q.
        Set to 1.0 if reserves are in foreign currency (they then revalue
        with e, and chi=1 under sticky prices maps to delta_A=1).
    """
    A_fin_0: float
    delta_A: float = 0.0

    def __post_init__(self) -> None:
        if self.A_fin_0 < 0:
            raise ValueError("A_fin_0 must be >= 0.")


@dataclass
class ElasticityParams:
    """
    Real-exchange-rate elasticity parameters for revenue and expenditure flows.

    [ASSUMPTION] Power-law response functions:
        X_t(q) = X_t0 * (q/q0)^eta_t

    These are calibrated parameters. The hypothesis beta_T > beta_N is
    maintained (tradable revenues more sensitive to q than nontradable),
    but is not enforced programmatically to allow calibration freedom.

    Parameters
    ----------
    beta_T : np.ndarray or float
        Elasticity of tradable revenue R_T_t w.r.t. q by period.
        Scalar or array of length T.
    beta_N : np.ndarray or float
        Elasticity of nontradable revenue R_N_t w.r.t. q by period.
    gamma_T : np.ndarray or float
        Elasticity of tradable expenditure G_T_t w.r.t. q by period.
    gamma_N : np.ndarray or float
        Elasticity of nontradable expenditure G_N_t w.r.t. q by period.
    """
    beta_T: np.ndarray
    beta_N: np.ndarray
    gamma_T: np.ndarray
    gamma_N: np.ndarray


@dataclass
class DiscountCurve:
    """
    Exogenous term structure of discount rates.

    [SOURCE] IMF TNM/10/02 (Escolano 2010) uses present-value discounting
    with a government borrowing rate. Rates are kept exogenous as instructed.

    Parameters
    ----------
    rates : np.ndarray
        Discount rates r_s for s = 1..T. Flat curve: pass np.full(T, r).
        Each rate must be > -1 (so discount factor is positive).
    """
    rates: np.ndarray            # shape (T,)

    def __post_init__(self) -> None:
        if np.any(self.rates <= -1.0):
            raise ValueError(
                "All discount rates must be > -1 (positive discount factors)."
            )
        if np.any(~np.isfinite(self.rates)):
            raise ValueError("Discount rates contain non-finite values.")

    def discount_factors(self) -> np.ndarray:
        """
        Compute cumulative discount factors D_t = prod_{s=1}^t (1 + r_s).

        Returns
        -------
        np.ndarray, shape (T,)
            D_t for t = 1..T.
        """
        return np.cumprod(1.0 + self.rates)


@dataclass
class SovereignSALMParams:
    """
    Master parameter container for the sovereign SALM model.

    Parameters
    ----------
    revenue_instruments : List[RevenueInstrument]
        All government revenue line items, GFSM-consistent.
    expenditure_instruments : List[ExpenditureInstrument]
        All government expenditure line items.
    debt : DebtBlock
        FX and LC debt parameters.
    fin_assets : FinancialAssets
        Current public financial assets.
    elasticities : ElasticityParams
        RER response elasticities for all flow blocks.
    discount_curve : DiscountCurve
        Exogenous term structure.
    q0 : float
        Baseline real exchange rate (higher = more depreciated).
    T : int
        Number of projection periods.
    """
    revenue_instruments: List[RevenueInstrument]
    expenditure_instruments: List[ExpenditureInstrument]
    debt: DebtBlock
    fin_assets: FinancialAssets
    elasticities: ElasticityParams
    discount_curve: DiscountCurve
    q0: float
    T: int

    def __post_init__(self) -> None:
        if self.q0 <= 0:
            raise ValueError("q0 must be strictly positive.")
        if self.T <= 0:
            raise ValueError("T must be a positive integer.")
        _T = self.T
        # Validate all array lengths
        for r in self.revenue_instruments:
            if len(r.amounts) != _T:
                raise ValueError(
                    f"RevenueInstrument '{r.name}': amounts length "
                    f"{len(r.amounts)} != T={_T}."
                )
        for g in self.expenditure_instruments:
            if len(g.amounts) != _T:
                raise ValueError(
                    f"ExpenditureInstrument '{g.name}': amounts length "
                    f"{len(g.amounts)} != T={_T}."
                )
        if len(self.discount_curve.rates) != _T:
            raise ValueError(
                f"Discount curve length {len(self.discount_curve.rates)} != T={_T}."
            )
        # Broadcast scalar elasticities to arrays
        self.elasticities.beta_T = _broadcast(
            self.elasticities.beta_T, _T, "beta_T"
        )
        self.elasticities.beta_N = _broadcast(
            self.elasticities.beta_N, _T, "beta_N"
        )
        self.elasticities.gamma_T = _broadcast(
            self.elasticities.gamma_T, _T, "gamma_T"
        )
        self.elasticities.gamma_N = _broadcast(
            self.elasticities.gamma_N, _T, "gamma_N"
        )


def _broadcast(x: float | np.ndarray, T: int, name: str) -> np.ndarray:
    """Broadcast scalar or 1-D array to length T."""
    arr = np.atleast_1d(np.asarray(x, dtype=float))
    if arr.ndim == 1 and len(arr) == 1:
        return np.full(T, arr[0])
    if len(arr) == T:
        return arr
    raise ValueError(f"Elasticity '{name}' has length {len(arr)}, expected 1 or {T}.")


# ============================================================================
# SECTION 2: ECONOMIC FORMULAS
# ============================================================================


def compute_RT_RN(
    instruments: List[RevenueInstrument],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose total revenues into tradable-linked (RT) and
    nontradable-linked (RN) components.

    [SOURCE] IMF GFSM 2014, Chapter 5; decomposition approach from
    IMF WP/12/241 sovereign balance sheet framework.

    [ASSUMPTION] alpha_i for each instrument is provided by the caller.
    If computed via exports/GDP proxy, that must be flagged on the instrument.

    Formula
    -------
    RT_t = sum_i alpha_i * R_{i,t}
    RN_t = sum_i (1 - alpha_i) * R_{i,t}

    Returns
    -------
    RT : np.ndarray, shape (T,)
    RN : np.ndarray, shape (T,)
    """
    if not instruments:
        raise ValueError("At least one revenue instrument required.")
    T = len(instruments[0].amounts)
    RT = np.zeros(T)
    RN = np.zeros(T)
    for inst in instruments:
        RT += inst.alpha * inst.amounts
        RN += (1.0 - inst.alpha) * inst.amounts
    return RT, RN


def compute_GT_GN(
    instruments: List[ExpenditureInstrument],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose total expenditures into tradable-linked (GT) and
    nontradable-linked (GN) components.

    [ASSUMPTION] phi_j for each instrument captures the share of expenditure
    whose real cost covaries positively with q (e.g., imported inputs,
    FX-denominated procurement, subsidies on tradable goods).

    Formula
    -------
    GT_t = sum_j phi_j * G_{j,t}
    GN_t = sum_j (1 - phi_j) * G_{j,t}

    Returns
    -------
    GT : np.ndarray, shape (T,)
    GN : np.ndarray, shape (T,)
    """
    if not instruments:
        raise ValueError("At least one expenditure instrument required.")
    T = len(instruments[0].amounts)
    GT = np.zeros(T)
    GN = np.zeros(T)
    for inst in instruments:
        GT += inst.phi * inst.amounts
        GN += (1.0 - inst.phi) * inst.amounts
    return GT, GN


def apply_q_response(
    baseline_flow: np.ndarray,
    q: float,
    q0: float,
    elasticities: np.ndarray,
) -> np.ndarray:
    """
    Apply the real-exchange-rate response function to a flow.

    [ASSUMPTION] Power-law specification:
        X_t(q) = X_t0 * (q/q0)^eta_t

    This is a parsimonious parametric form. For eta=0, the flow is
    invariant to q. For eta=1, the flow moves proportionally with q.

    Parameters
    ----------
    baseline_flow : np.ndarray, shape (T,)
        Flow at baseline q0.
    q : float
        Shocked real exchange rate.
    q0 : float
        Baseline real exchange rate.
    elasticities : np.ndarray, shape (T,)
        Period-specific elasticity eta_t.

    Returns
    -------
    np.ndarray, shape (T,)
    """
    if q <= 0 or q0 <= 0:
        raise ValueError("q and q0 must be strictly positive.")
    ratio = q / q0
    return baseline_flow * (ratio ** elasticities)


def compute_D_FX(
    D_FX_0: float,
    q: float,
    q0: float,
    chi: float,
    psi_q_to_e: float = 1.0,
) -> float:
    """
    Compute the domestic-currency value of FX debt at real exchange rate q.

    [ASSUMPTION]
    FX debt is fundamentally revalued by the nominal exchange rate e. When the
    model is expressed as a function of q, the effective q-elasticity is:

        eta_DFX,q = chi * psi_q_to_e

    where:
        chi        = d ln(D_FX) / d ln(e)
        psi_q_to_e = d ln(e) / d ln(q)

    Therefore:
        D_FX(q) = D_FX_0 * (q/q0)^(chi * psi_q_to_e)

    Under the short-horizon sticky-price baseline:
        psi_q_to_e = 1
    so the q-shock and the e-shock coincide locally.

    [SOURCE] IMF real-exchange-rate definition; BIS EER methodology; IMF SALM
    framework for sovereign balance-sheet valuation.

    Parameters
    ----------
    D_FX_0 : float
        FX debt at baseline q0 in domestic-currency equivalent.
    q : float
        Shocked real exchange rate.
    q0 : float
        Baseline real exchange rate.
    chi : float
        Elasticity of D_FX with respect to the nominal exchange rate e.
    psi_q_to_e : float
        Mapping from q shocks to e shocks.

    Returns
    -------
    float
    """
    return D_FX_0 * (q / q0) ** (chi * psi_q_to_e)


def compute_pv(
    flows: np.ndarray,
    discount_factors: np.ndarray,
) -> float:
    """
    Compute the present value of a flow series.

    [SOURCE] IMF TNM/10/02 (Escolano 2010), standard PV discounting.

    Formula
    -------
    PV(X) = sum_{t=1}^T X_t / prod_{s=1}^t (1 + r_s)
           = sum_{t=1}^T X_t / D_t

    where D_t = prod_{s=1}^t (1 + r_s) are cumulative discount factors.

    Parameters
    ----------
    flows : np.ndarray, shape (T,)
        Nominal flow amounts by period.
    discount_factors : np.ndarray, shape (T,)
        Cumulative discount factors D_t > 0.

    Returns
    -------
    float
        Present value.
    """
    if np.any(discount_factors <= 0):
        raise ValueError("All cumulative discount factors must be > 0.")
    pv = float(np.sum(flows / discount_factors))
    if not math.isfinite(pv):
        raise ArithmeticError(f"PV computation yielded non-finite result: {pv}.")
    return pv


# ============================================================================
# SECTION 3: CORE MODEL — BALANCE SHEET COMPUTATION
# ============================================================================


@dataclass
class BalanceSheet:
    """
    Sovereign economic balance sheet at a given real exchange rate q.

    [SOURCE] IMF WP/12/241 (2012), IMF WP/19/290 (2019).

    Fields
    ------
    q : float
        Real exchange rate at which the balance sheet is evaluated.
    A_fin : float
        Current public financial assets (revalued at q).
    PV_RT : float
        Present value of tradable-linked revenues.
    PV_RN : float
        Present value of nontradable-linked revenues.
    A_total : float
        Total sovereign assets = A_fin + PV_RT + PV_RN.
    D_FX : float
        FX debt revalued at q (domestic currency).
    D_LC : float
        Local-currency debt (invariant to q in baseline).
    PV_GT : float
        Present value of tradable-linked expenditures.
    PV_GN : float
        Present value of nontradable-linked expenditures.
    L_total : float
        Total sovereign liabilities = D_FX + D_LC + PV_GT + PV_GN.
    leverage : float
        lambda(q) = L_total / A_total.
    """
    q: float
    A_fin: float
    PV_RT: float
    PV_RN: float
    A_total: float
    D_FX: float
    D_LC: float
    PV_GT: float
    PV_GN: float
    L_total: float
    leverage: float

    def summary(self) -> pd.DataFrame:
        """Return a tidy DataFrame representation of the balance sheet."""
        assets = {
            "Financial assets (A_fin)": self.A_fin,
            "PV(Tradable revenue R_T)": self.PV_RT,
            "PV(Nontradable revenue R_N)": self.PV_RN,
            "TOTAL ASSETS": self.A_total,
        }
        liabilities = {
            "FX Debt (D_FX at q)": self.D_FX,
            "LC Debt (D_LC)": self.D_LC,
            "PV(Tradable expenditure G_T)": self.PV_GT,
            "PV(Nontradable expenditure G_N)": self.PV_GN,
            "TOTAL LIABILITIES": self.L_total,
        }
        rows = []
        for k, v in assets.items():
            rows.append({"Side": "Assets", "Item": k, "Value": v})
        for k, v in liabilities.items():
            rows.append({"Side": "Liabilities", "Item": k, "Value": v})
        rows.append({"Side": "—", "Item": "LEVERAGE lambda(q)", "Value": self.leverage})
        return pd.DataFrame(rows).set_index(["Side", "Item"])


def compute_balance_sheet(
    params: SovereignSALMParams,
    q: float,
) -> BalanceSheet:
    """
    Compute the sovereign economic balance sheet at real exchange rate q.

    Parameters
    ----------
    params : SovereignSALMParams
        Full model parameterization.
    q : float
        Real exchange rate (q > 0; higher = more depreciated).

    Returns
    -------
    BalanceSheet
    """
    if q <= 0:
        raise ValueError("q must be strictly positive.")

    p = params
    q0 = p.q0
    E = p.elasticities
    D = p.discount_curve.discount_factors()

    # ------------------------------------------------------------------
    # STEP 1: Baseline flow decompositions
    # ------------------------------------------------------------------
    RT0, RN0 = compute_RT_RN(p.revenue_instruments)
    GT0, GN0 = compute_GT_GN(p.expenditure_instruments)

    # ------------------------------------------------------------------
    # STEP 2: Apply RER response functions
    # ------------------------------------------------------------------
    RT_q = apply_q_response(RT0, q, q0, E.beta_T)
    RN_q = apply_q_response(RN0, q, q0, E.beta_N)
    GT_q = apply_q_response(GT0, q, q0, E.gamma_T)
    GN_q = apply_q_response(GN0, q, q0, E.gamma_N)

    # ------------------------------------------------------------------
    # STEP 3: Present values
    # ------------------------------------------------------------------
    pv_RT = compute_pv(RT_q, D)
    pv_RN = compute_pv(RN_q, D)
    pv_GT = compute_pv(GT_q, D)
    pv_GN = compute_pv(GN_q, D)

    # ------------------------------------------------------------------
    # STEP 4: Financial assets (revalued at q)
    # ------------------------------------------------------------------
    a_fin = p.fin_assets.A_fin_0 * (q / q0) ** p.fin_assets.delta_A

    # ------------------------------------------------------------------
    # STEP 5: FX debt (revalued at q)
    # ------------------------------------------------------------------
    d_fx = compute_D_FX(p.debt.D_FX_0, q, q0, p.debt.chi, p.debt.psi_q_to_e)

    # ------------------------------------------------------------------
    # STEP 6: Totals and leverage
    # ------------------------------------------------------------------
    A = a_fin + pv_RT + pv_RN
    L = d_fx + p.debt.D_LC + pv_GT + pv_GN

    if A <= 0:
        raise ArithmeticError(
            f"Total assets A={A:.6g} <= 0 at q={q}. "
            "Balance sheet is technically insolvent or miscalibrated."
        )

    lam = L / A

    return BalanceSheet(
        q=q,
        A_fin=a_fin,
        PV_RT=pv_RT,
        PV_RN=pv_RN,
        A_total=A,
        D_FX=d_fx,
        D_LC=p.debt.D_LC,
        PV_GT=pv_GT,
        PV_GN=pv_GN,
        L_total=L,
        leverage=lam,
    )


# ============================================================================
# SECTION 4: ELASTICITY ROUTINES
# ============================================================================


@dataclass
class ElasticityResult:
    """
    Output of the leverage-ratio elasticity calculation.

    Fields
    ------
    epsilon_analytical : float
        Analytical elasticity d ln(lambda) / d ln(q) from weighted formula.
    epsilon_numerical : float
        Numerical central-difference approximation.
    match : bool
        True if |analytical - numerical| <= tolerance.
    tolerance : float
        Tolerance used for the cross-check.
    weight_DFX : float      Weight of D_FX in total liabilities.
    weight_DLC : float      Weight of D_LC in total liabilities.
    weight_GT  : float      Weight of PV(G_T) in total liabilities.
    weight_GN  : float      Weight of PV(G_N) in total liabilities.
    weight_Afin: float      Weight of A_fin in total assets.
    weight_RT  : float      Weight of PV(R_T) in total assets.
    weight_RN  : float      Weight of PV(R_N) in total assets.
    eta_DFX    : float      Elasticity of D_FX w.r.t. q.
    eta_DLC    : float      Elasticity of D_LC w.r.t. q (= 0 baseline).
    eta_GT     : float      PV-weighted elasticity of G_T block.
    eta_GN     : float      PV-weighted elasticity of G_N block.
    eta_Afin   : float      Elasticity of A_fin w.r.t. q.
    eta_RT     : float      PV-weighted elasticity of R_T block.
    eta_RN     : float      PV-weighted elasticity of R_N block.
    d_ln_L_d_ln_q : float  Liability log-elasticity.
    d_ln_A_d_ln_q : float  Asset log-elasticity.
    """
    epsilon_analytical: float
    epsilon_numerical: float
    match: bool
    tolerance: float
    weight_DFX: float
    weight_DLC: float
    weight_GT: float
    weight_GN: float
    weight_Afin: float
    weight_RT: float
    weight_RN: float
    eta_DFX: float
    eta_DLC: float
    eta_GT: float
    eta_GN: float
    eta_Afin: float
    eta_RT: float
    eta_RN: float
    d_ln_L_d_ln_q: float
    d_ln_A_d_ln_q: float


def _pv_weighted_elasticity(
    baseline_flow: np.ndarray,
    elasticities: np.ndarray,
    discount_factors: np.ndarray,
) -> float:
    """
    Compute the PV-weighted average elasticity of a flow block.

    [DERIVATION]
    If X_t(q) = X_t0 * (q/q0)^eta_t, then
        d ln PV(X) / d ln q = sum_t [ X_t0 / D_t * eta_t ] / sum_t [ X_t0 / D_t ]
                            = sum_t omega_t * eta_t

    where omega_t = (X_t0/D_t) / PV(X) is the period-t present-value weight.

    This is exact when discount factors are invariant to q (exogenous discount
    curve assumption per modeling instructions).

    Returns
    -------
    float : PV-weighted average of eta_t.
    """
    pv_weights = baseline_flow / discount_factors   # numerator by period
    total = pv_weights.sum()
    if total == 0.0:
        return 0.0                                   # zero flow => zero elasticity
    return float(np.dot(pv_weights, elasticities) / total)


def compute_elasticity_analytical(
    params: SovereignSALMParams,
    bs: BalanceSheet,
) -> ElasticityResult:
    """
    Compute the leverage elasticity analytically using the weighted-elasticity
    decomposition formula.

    [DERIVATION]
    epsilon_{lambda,q} = d ln L / d ln q - d ln A / d ln q

    d ln L / d ln q = w_DFX * eta_DFX + w_DLC * eta_DLC
                    + w_GT * eta_GT_q + w_GN * eta_GN_q

    d ln A / d ln q = v_Afin * eta_Afin + v_RT * eta_RT_q + v_RN * eta_RN_q

    where w_k = X_k / L and v_m = X_m / A are value-weights at baseline q0.

    [IMPL] All weights and elasticities are evaluated at q = q0 (the baseline)
    to obtain the local elasticity. For large shocks, numerical approximation
    is more reliable.

    Parameters
    ----------
    params : SovereignSALMParams
    bs : BalanceSheet
        Balance sheet evaluated at q = q0 (baseline).
    """
    p = params
    E = p.elasticities
    D = p.discount_curve.discount_factors()

    # Baseline flow decompositions (at q0)
    RT0, RN0 = compute_RT_RN(p.revenue_instruments)
    GT0, GN0 = compute_GT_GN(p.expenditure_instruments)

    # ------------------------------------------------------------------
    # PV-weighted elasticities for each flow block
    # ------------------------------------------------------------------
    eta_RT = _pv_weighted_elasticity(RT0, E.beta_T, D)
    eta_RN = _pv_weighted_elasticity(RN0, E.beta_N, D)
    eta_GT = _pv_weighted_elasticity(GT0, E.gamma_T, D)
    eta_GN = _pv_weighted_elasticity(GN0, E.gamma_N, D)

    # Stock block elasticities
    eta_DFX = p.debt.eta_DFX_q    # = chi * psi_q_to_e; sticky-price baseline => 1
    eta_DLC = 0.0                 # LC debt invariant to q in baseline
    eta_Afin = p.fin_assets.delta_A

    # ------------------------------------------------------------------
    # Value weights
    # ------------------------------------------------------------------
    L = bs.L_total
    A = bs.A_total

    w_DFX = bs.D_FX / L
    w_DLC = bs.D_LC / L
    w_GT  = bs.PV_GT / L
    w_GN  = bs.PV_GN / L

    v_Afin = bs.A_fin / A
    v_RT   = bs.PV_RT / A
    v_RN   = bs.PV_RN / A

    # ------------------------------------------------------------------
    # Weighted log-elasticities of L and A
    # ------------------------------------------------------------------
    d_ln_L = w_DFX * eta_DFX + w_DLC * eta_DLC + w_GT * eta_GT + w_GN * eta_GN
    d_ln_A = v_Afin * eta_Afin + v_RT * eta_RT + v_RN * eta_RN

    epsilon_analytical = d_ln_L - d_ln_A

    # ------------------------------------------------------------------
    # Numerical cross-check (central difference around q0)
    # ------------------------------------------------------------------
    epsilon_numerical = _numerical_elasticity(params, p.q0)

    tolerance = 1e-6
    match = abs(epsilon_analytical - epsilon_numerical) <= tolerance

    if not match:
        warnings.warn(
            f"Analytical elasticity {epsilon_analytical:.8f} differs from "
            f"numerical {epsilon_numerical:.8f} by "
            f"{abs(epsilon_analytical - epsilon_numerical):.2e} "
            f"(tolerance {tolerance:.1e}). Check elasticity parameters.",
            UserWarning,
            stacklevel=2,
        )

    return ElasticityResult(
        epsilon_analytical=epsilon_analytical,
        epsilon_numerical=epsilon_numerical,
        match=match,
        tolerance=tolerance,
        weight_DFX=w_DFX,
        weight_DLC=w_DLC,
        weight_GT=w_GT,
        weight_GN=w_GN,
        weight_Afin=v_Afin,
        weight_RT=v_RT,
        weight_RN=v_RN,
        eta_DFX=eta_DFX,
        eta_DLC=eta_DLC,
        eta_GT=eta_GT,
        eta_GN=eta_GN,
        eta_Afin=eta_Afin,
        eta_RT=eta_RT,
        eta_RN=eta_RN,
        d_ln_L_d_ln_q=d_ln_L,
        d_ln_A_d_ln_q=d_ln_A,
    )


def _numerical_elasticity(
    params: SovereignSALMParams,
    q: float,
    h: float = 1e-5,
) -> float:
    """
    Numerical central-difference approximation of d ln lambda / d ln q.

    [IMPL] Central-difference formula:
        d ln f / d ln q ≈ [ln f(q*(1+h)) - ln f(q*(1-h))] / (2h)
    for small h. This is O(h^2) accurate.

    Parameters
    ----------
    params : SovereignSALMParams
    q : float
        Point at which to evaluate the elasticity.
    h : float
        Relative step size (default 1e-5).

    Returns
    -------
    float
    """
    q_up   = q * (1.0 + h)
    q_down = q * (1.0 - h)
    bs_up   = compute_balance_sheet(params, q_up)
    bs_down = compute_balance_sheet(params, q_down)
    lam_up   = bs_up.leverage
    lam_down = bs_down.leverage
    return (math.log(lam_up) - math.log(lam_down)) / (2.0 * h)


# ============================================================================
# SECTION 5: CALIBRATION HELPERS
# ============================================================================


def alpha_from_exports_gdp(
    exports_gdp_ratio: float,
    instrument_name: str,
) -> Tuple[float, bool]:
    """
    Compute the tradable-basis share alpha for a revenue instrument using
    the exports/GDP ratio as a proxy.

    [ASSUMPTION — USER-APPROVED FALLBACK]
    This is an approximation, not a theoretical identity. It is appropriate
    only when instrument-level base decomposition is unavailable.
    The exports/GDP ratio proxies the share of economic activity that is
    tradable-sector-linked.

    Source logic: Calvo, Izquierdo, and Talvi (2003, NBER WP 9828) use exports
    as a proxy for tradable output "particularly in the short run," acknowledging
    it may underestimate tradable output in the long run.

    Parameters
    ----------
    exports_gdp_ratio : float
        Exports / GDP, in [0, 1].
    instrument_name : str
        Name of the revenue instrument (for logging).

    Returns
    -------
    alpha : float
    use_proxy : bool  (always True when this function is used)
    """
    if not (0.0 <= exports_gdp_ratio <= 1.0):
        raise ValueError(
            f"exports_gdp_ratio={exports_gdp_ratio} must be in [0, 1]."
        )
    warnings.warn(
        f"Instrument '{instrument_name}': alpha set to exports/GDP ratio "
        f"({exports_gdp_ratio:.4f}). This is an approximation, not an "
        "identity. Use instrument-specific revenue-base analysis when available.",
        UserWarning,
        stacklevel=2,
    )
    return exports_gdp_ratio, True


def build_flat_discount_curve(r: float, T: int) -> DiscountCurve:
    """
    Build a flat discount curve with constant rate r over T periods.

    Parameters
    ----------
    r : float
        Annual discount rate (e.g., 0.05 for 5%).
    T : int
        Number of periods.

    Returns
    -------
    DiscountCurve
    """
    return DiscountCurve(rates=np.full(T, r))


def load_params_from_dataframes(
    revenue_df: pd.DataFrame,
    expenditure_df: pd.DataFrame,
    macro_scalars: Dict[str, float],
    elasticity_scalars: Dict[str, float],
    discount_rates: np.ndarray,
    T: int,
) -> SovereignSALMParams:
    """
    Construct SovereignSALMParams from tidy DataFrames.

    [IMPL] This is the recommended I/O path for loading from CSV or Excel.

    revenue_df columns:
        name, amount_t1, amount_t2, ..., amount_tT, alpha, use_proxy

    expenditure_df columns:
        name, amount_t1, amount_t2, ..., amount_tT, phi

    macro_scalars keys:
        D_FX_0, D_LC, A_fin_0, delta_A, chi, psi_q_to_e, q0

    elasticity_scalars keys:
        beta_T, beta_N, gamma_T, gamma_N
        (scalar or list of T values — passed as comma-separated string if from CSV)

    Parameters
    ----------
    revenue_df : pd.DataFrame
    expenditure_df : pd.DataFrame
    macro_scalars : Dict[str, float]
    elasticity_scalars : Dict[str, float]  values can be float or list
    discount_rates : np.ndarray of length T
    T : int

    Returns
    -------
    SovereignSALMParams
    """
    def _parse_amounts(row: pd.Series, T: int) -> np.ndarray:
        cols = [f"amount_t{t+1}" for t in range(T)]
        return row[cols].to_numpy(dtype=float)

    rev_instruments = []
    for _, row in revenue_df.iterrows():
        rev_instruments.append(
            RevenueInstrument(
                name=str(row["name"]),
                amounts=_parse_amounts(row, T),
                alpha=float(row["alpha"]),
                use_proxy=bool(row.get("use_proxy", False)),
            )
        )

    exp_instruments = []
    for _, row in expenditure_df.iterrows():
        exp_instruments.append(
            ExpenditureInstrument(
                name=str(row["name"]),
                amounts=_parse_amounts(row, T),
                phi=float(row["phi"]),
            )
        )

    def _parse_elast(v) -> np.ndarray:
        if isinstance(v, (int, float)):
            return np.full(T, float(v))
        if isinstance(v, (list, np.ndarray)):
            return np.asarray(v, dtype=float)
        raise TypeError(f"Unrecognised elasticity type: {type(v)}")

    elast = ElasticityParams(
        beta_T=_parse_elast(elasticity_scalars["beta_T"]),
        beta_N=_parse_elast(elasticity_scalars["beta_N"]),
        gamma_T=_parse_elast(elasticity_scalars["gamma_T"]),
        gamma_N=_parse_elast(elasticity_scalars["gamma_N"]),
    )

    debt = DebtBlock(
        D_FX_0=macro_scalars["D_FX_0"],
        D_LC=macro_scalars["D_LC"],
        chi=macro_scalars.get("chi", 1.0),
        psi_q_to_e=macro_scalars.get("psi_q_to_e", 1.0),
    )

    fin_assets = FinancialAssets(
        A_fin_0=macro_scalars["A_fin_0"],
        delta_A=macro_scalars.get("delta_A", 0.0),
    )

    return SovereignSALMParams(
        revenue_instruments=rev_instruments,
        expenditure_instruments=exp_instruments,
        debt=debt,
        fin_assets=fin_assets,
        elasticities=elast,
        discount_curve=DiscountCurve(rates=discount_rates),
        q0=macro_scalars["q0"],
        T=T,
    )


# ============================================================================
# SECTION 6: REPORTING / OUTPUT LAYER
# ============================================================================


def scenario_analysis(
    params: SovereignSALMParams,
    q_values: List[float],
) -> pd.DataFrame:
    """
    Compute balance sheet and elasticity across a grid of q values.

    Parameters
    ----------
    params : SovereignSALMParams
    q_values : List[float]
        Grid of real exchange rates to evaluate.

    Returns
    -------
    pd.DataFrame with columns:
        q, A_fin, PV_RT, PV_RN, A_total,
        D_FX, D_LC, PV_GT, PV_GN, L_total,
        leverage, epsilon_numerical
    """
    rows = []
    for q in q_values:
        bs = compute_balance_sheet(params, q)
        eps_num = _numerical_elasticity(params, q)
        rows.append({
            "q": bs.q,
            "A_fin": bs.A_fin,
            "PV_RT": bs.PV_RT,
            "PV_RN": bs.PV_RN,
            "A_total": bs.A_total,
            "D_FX": bs.D_FX,
            "D_LC": bs.D_LC,
            "PV_GT": bs.PV_GT,
            "PV_GN": bs.PV_GN,
            "L_total": bs.L_total,
            "leverage": bs.leverage,
            "epsilon_numerical": eps_num,
        })
    return pd.DataFrame(rows)


def print_elasticity_decomposition(er: ElasticityResult) -> None:
    """Pretty-print the elasticity decomposition."""
    print("=" * 65)
    print("LEVERAGE ELASTICITY DECOMPOSITION  d ln(λ) / d ln(q)")
    print("=" * 65)

    print("\n  LIABILITY SIDE  [d ln L / d ln q = Σ w_k * η_k]")
    print(f"    FX Debt        w={er.weight_DFX:.4f}  η={er.eta_DFX:.4f}  "
          f"contribution={er.weight_DFX * er.eta_DFX:.4f}")
    print(f"    LC Debt        w={er.weight_DLC:.4f}  η={er.eta_DLC:.4f}  "
          f"contribution={er.weight_DLC * er.eta_DLC:.4f}")
    print(f"    PV(G_T)        w={er.weight_GT:.4f}  η={er.eta_GT:.4f}  "
          f"contribution={er.weight_GT * er.eta_GT:.4f}")
    print(f"    PV(G_N)        w={er.weight_GN:.4f}  η={er.eta_GN:.4f}  "
          f"contribution={er.weight_GN * er.eta_GN:.4f}")
    print(f"    TOTAL          d ln L / d ln q = {er.d_ln_L_d_ln_q:.6f}")

    print("\n  ASSET SIDE  [d ln A / d ln q = Σ v_m * η_m]")
    print(f"    A_fin          v={er.weight_Afin:.4f}  η={er.eta_Afin:.4f}  "
          f"contribution={er.weight_Afin * er.eta_Afin:.4f}")
    print(f"    PV(R_T)        v={er.weight_RT:.4f}  η={er.eta_RT:.4f}  "
          f"contribution={er.weight_RT * er.eta_RT:.4f}")
    print(f"    PV(R_N)        v={er.weight_RN:.4f}  η={er.eta_RN:.4f}  "
          f"contribution={er.weight_RN * er.eta_RN:.4f}")
    print(f"    TOTAL          d ln A / d ln q = {er.d_ln_A_d_ln_q:.6f}")

    print("\n  LEVERAGE ELASTICITY")
    print(f"    Analytical  ε_{{λ,q}} = {er.epsilon_analytical:.8f}")
    print(f"    Numerical   ε_{{λ,q}} = {er.epsilon_numerical:.8f}")
    print(f"    Match within tol={er.tolerance:.0e}: {er.match}")
    print("=" * 65)


# ============================================================================
# SECTION 7: UNIT TESTS
# ============================================================================


def run_all_tests(verbose: bool = True) -> None:
    """
    Run all unit tests. Raises AssertionError on failure.

    Tests
    -----
    1. Decomposition identity: RT + RN == total R, GT + GN == total G
    2. PV calculation (flat rate, single period)
    3. Analytical vs numerical elasticity cross-check
    4. Sticky-price equivalence: ε_{λ,q} = ε_{λ,e} when chi=1, fixed P/P*
    5. Zero FX debt case: D_FX = 0 → no FX liability contribution
    6. Zero tradable revenue: beta_T has no effect when RT = 0
    7. All-LC-liability case: lambda insensitive to q if DFX=0, GT=GN=0
    """
    print("Running unit tests...\n")

    # ----------------------------------------------------------------
    # TEST 1: Decomposition identity
    # ----------------------------------------------------------------
    _test_decomposition_identity(verbose)

    # ----------------------------------------------------------------
    # TEST 2: PV calculation
    # ----------------------------------------------------------------
    _test_pv_flat_rate(verbose)

    # ----------------------------------------------------------------
    # TEST 3: Analytical vs numerical elasticity
    # ----------------------------------------------------------------
    _test_analytical_vs_numerical(verbose)

    # ----------------------------------------------------------------
    # TEST 4: Sticky-price equivalence
    # ----------------------------------------------------------------
    _test_sticky_price_equivalence(verbose)

    # ----------------------------------------------------------------
    # TEST 5: Zero FX debt
    # ----------------------------------------------------------------
    _test_zero_fx_debt(verbose)

    # ----------------------------------------------------------------
    # TEST 6: Zero tradable revenue (RT=0)
    # ----------------------------------------------------------------
    _test_zero_tradable_revenue(verbose)

    # ----------------------------------------------------------------
    # TEST 7: All LC liabilities
    # ----------------------------------------------------------------
    _test_all_lc_liabilities(verbose)

    print("\nAll tests passed.")


def _make_minimal_params(
    T: int = 5,
    alpha: float = 0.3,
    phi: float = 0.2,
    D_FX_0: float = 100.0,
    D_LC: float = 200.0,
    A_fin_0: float = 50.0,
    delta_A: float = 0.0,
    chi: float = 1.0,
    psi_q_to_e: float = 1.0,
    beta_T: float = 0.8,
    beta_N: float = 0.1,
    gamma_T: float = 0.5,
    gamma_N: float = 0.05,
    r: float = 0.05,
    q0: float = 1.0,
    rev_amounts: Optional[np.ndarray] = None,
    exp_amounts: Optional[np.ndarray] = None,
) -> SovereignSALMParams:
    """Helper: build a minimal but non-trivial parameter set."""
    if rev_amounts is None:
        rev_amounts = np.full(T, 80.0)
    if exp_amounts is None:
        exp_amounts = np.full(T, 60.0)

    return SovereignSALMParams(
        revenue_instruments=[
            RevenueInstrument("total_revenue", rev_amounts.copy(), alpha)
        ],
        expenditure_instruments=[
            ExpenditureInstrument("total_expenditure", exp_amounts.copy(), phi)
        ],
        debt=DebtBlock(D_FX_0=D_FX_0, D_LC=D_LC, chi=chi, psi_q_to_e=psi_q_to_e),
        fin_assets=FinancialAssets(A_fin_0=A_fin_0, delta_A=delta_A),
        elasticities=ElasticityParams(
            beta_T=np.full(T, beta_T),
            beta_N=np.full(T, beta_N),
            gamma_T=np.full(T, gamma_T),
            gamma_N=np.full(T, gamma_N),
        ),
        discount_curve=build_flat_discount_curve(r, T),
        q0=q0,
        T=T,
    )


def _test_decomposition_identity(verbose: bool) -> None:
    """RT + RN must equal total revenue by construction."""
    T = 4
    instruments = [
        RevenueInstrument("vat",      np.array([100.0, 105.0, 110.0, 115.0]), alpha=0.25),
        RevenueInstrument("customs",  np.array([30.0,  31.0,  32.0,  33.0]),  alpha=0.95),
        RevenueInstrument("corp_tax", np.array([50.0,  52.0,  54.0,  56.0]),  alpha=0.40),
    ]
    total = sum(inst.amounts for inst in instruments)
    RT, RN = compute_RT_RN(instruments)
    assert np.allclose(RT + RN, total, atol=1e-12), \
        f"Decomposition identity failed: RT+RN != total R"

    exp_instruments = [
        ExpenditureInstrument("wages",   np.array([80.0, 82.0, 84.0, 86.0]), phi=0.05),
        ExpenditureInstrument("imports", np.array([20.0, 21.0, 22.0, 23.0]), phi=0.90),
    ]
    total_g = sum(inst.amounts for inst in exp_instruments)
    GT, GN = compute_GT_GN(exp_instruments)
    assert np.allclose(GT + GN, total_g, atol=1e-12), \
        "Decomposition identity failed: GT+GN != total G"

    if verbose:
        print("[PASS] Test 1: Decomposition identity RT+RN=R, GT+GN=G")


def _test_pv_flat_rate(verbose: bool) -> None:
    """PV with flat rate r=10%, single period: PV = X/(1+r)."""
    flows = np.array([110.0])
    D = np.array([1.1])
    pv = compute_pv(flows, D)
    assert abs(pv - 100.0) < 1e-10, f"PV test failed: {pv} != 100.0"

    # Multi-period: annuity of 100 for 3 periods at 5%
    flows3 = np.array([100.0, 100.0, 100.0])
    r = 0.05
    D3 = np.cumprod(np.full(3, 1 + r))
    pv3 = compute_pv(flows3, D3)
    pv3_expected = sum(100.0 / (1.05 ** t) for t in range(1, 4))
    assert abs(pv3 - pv3_expected) < 1e-10, f"PV annuity test failed: {pv3}"

    if verbose:
        print("[PASS] Test 2: PV calculation (single period + annuity)")


def _test_analytical_vs_numerical(verbose: bool) -> None:
    """Analytical and numerical elasticities must agree within 1e-6."""
    p = _make_minimal_params()
    bs0 = compute_balance_sheet(p, p.q0)
    er = compute_elasticity_analytical(p, bs0)
    assert er.match, (
        f"Analytical ({er.epsilon_analytical:.8f}) vs "
        f"numerical ({er.epsilon_numerical:.8f}) mismatch."
    )
    if verbose:
        print(f"[PASS] Test 3: Analytical ε={er.epsilon_analytical:.8f} "
              f"≈ Numerical ε={er.epsilon_numerical:.8f}")


def _test_sticky_price_equivalence(verbose: bool) -> None:
    """
    Under sticky prices, psi_q_to_e = 1 so the effective q-elasticity of
    FX debt equals its nominal-FX elasticity: eta_DFX_q = chi.
    Verify the model respects this mapping.
    """
    # With chi=1 and psi_q_to_e=1, the FX debt block loads 1-for-1 on q.
    p = _make_minimal_params(chi=1.0)
    bs0 = compute_balance_sheet(p, p.q0)
    er = compute_elasticity_analytical(p, bs0)

    # eta_DFX_q must equal chi when psi_q_to_e = 1
    assert abs(er.eta_DFX - 1.0) < 1e-12,         f"eta_DFX={er.eta_DFX} != chi*psi_q_to_e = 1"

    # Verify that changing the q->e mapping changes the effective q-elasticity
    p2 = _make_minimal_params(chi=1.0, psi_q_to_e=0.5)
    bs02 = compute_balance_sheet(p2, p2.q0)
    er2 = compute_elasticity_analytical(p2, bs02)
    assert abs(er2.eta_DFX - 0.5) < 1e-12,         f"eta_DFX={er2.eta_DFX} != chi*psi_q_to_e = 0.5"

    if verbose:
        print(f"[PASS] Test 4: Sticky-price mapping "
              f"η_DFX_q(psi=1)={er.eta_DFX:.6f}, η_DFX_q(psi=0.5)={er2.eta_DFX:.6f}")


def _test_zero_fx_debt(verbose: bool) -> None:
    """With D_FX_0=0, FX debt contributes nothing to the liability elasticity."""
    p = _make_minimal_params(D_FX_0=0.0)
    bs0 = compute_balance_sheet(p, p.q0)
    er = compute_elasticity_analytical(p, bs0)

    # FX weight must be zero
    assert abs(er.weight_DFX) < 1e-12, \
        f"weight_DFX={er.weight_DFX} != 0 when D_FX_0=0"

    # D_FX at any q must also be zero
    d_fx_shocked = compute_D_FX(0.0, 1.5, 1.0, 1.0)
    assert d_fx_shocked == 0.0

    if verbose:
        print(f"[PASS] Test 5: Zero FX debt — w_DFX={er.weight_DFX:.6f}")


def _test_zero_tradable_revenue(verbose: bool) -> None:
    """
    If all alpha_i = 0, then RT = 0 and PV(RT) = 0.
    beta_T becomes irrelevant; the model reduces to RN only.
    """
    p = _make_minimal_params(alpha=0.0)
    bs0 = compute_balance_sheet(p, p.q0)

    assert abs(bs0.PV_RT) < 1e-12, \
        f"PV_RT={bs0.PV_RT} != 0 when alpha=0"

    # Asset elasticity should have no RT contribution
    er = compute_elasticity_analytical(p, bs0)
    assert abs(er.weight_RT) < 1e-12, \
        f"weight_RT={er.weight_RT} != 0 when RT=0"

    if verbose:
        print(f"[PASS] Test 6: Zero tradable revenue — PV_RT={bs0.PV_RT:.6f}, "
              f"w_RT={er.weight_RT:.6f}")


def _test_all_lc_liabilities(verbose: bool) -> None:
    """
    If D_FX=0, GT=0, GN=0, liabilities are only D_LC (invariant to q).
    The liability elasticity must be zero.
    """
    p = _make_minimal_params(
        D_FX_0=0.0,
        D_LC=500.0,
        phi=0.0,              # all expenditure is nontradable
        gamma_T=0.0,
        gamma_N=0.0,          # expenditure does not move with q
        exp_amounts=np.zeros(5),  # zero expenditure => PV(GT)=PV(GN)=0
    )
    bs0 = compute_balance_sheet(p, p.q0)
    er = compute_elasticity_analytical(p, bs0)

    assert abs(er.d_ln_L_d_ln_q) < 1e-10, \
        f"d ln L / d ln q = {er.d_ln_L_d_ln_q} != 0 for all-LC case"

    if verbose:
        print(f"[PASS] Test 7: All-LC liabilities — "
              f"d ln L / d ln q = {er.d_ln_L_d_ln_q:.8f}")


# ============================================================================
# SECTION 8: WORKED EXAMPLE
# ============================================================================


def worked_example() -> None:
    """
    Reproducible worked example: small open economy with commodity exports.

    CONTEXT
    -------
    A stylised EME with:
    - 5-year projection horizon
    - Revenue from VAT, corporate income tax (CIT), and resource royalties
    - Expenditure on wages/transfers and imported capital goods
    - External FX debt and local LC bonds
    - Exports/GDP ≈ 30% used as proxy alpha for VAT and CIT (user-approved)
    - Royalties treated as 95% tradable (resource revenues priced in USD)

    ALL PARAMETERS ARE ILLUSTRATIVE AND DO NOT REPRESENT ANY REAL COUNTRY.
    """
    print("\n" + "=" * 65)
    print("WORKED EXAMPLE: Stylised Commodity-Exporting EME")
    print("=" * 65)

    T = 5
    q0 = 1.0

    # ------------------------------------------------------------------
    # Revenue instruments (amounts in millions of domestic currency, DCU)
    # ------------------------------------------------------------------
    # Alpha for VAT and CIT: use exports/GDP proxy (user-approved fallback)
    exports_gdp = 0.30
    alpha_vat, proxy_flag_vat = alpha_from_exports_gdp(exports_gdp, "VAT")
    alpha_cit, proxy_flag_cit = alpha_from_exports_gdp(exports_gdp, "CIT")

    # Royalties are essentially a tradable revenue: priced at world commodity
    # prices in USD. Alpha set to 0.95 by direct revenue-base analysis.
    alpha_royalties = 0.95

    rev_instruments = [
        RevenueInstrument(
            name="VAT",
            amounts=np.array([400.0, 412.0, 424.0, 437.0, 450.0]),
            alpha=alpha_vat,
            use_proxy=proxy_flag_vat,
        ),
        RevenueInstrument(
            name="CIT",
            amounts=np.array([150.0, 155.0, 160.0, 165.0, 170.0]),
            alpha=alpha_cit,
            use_proxy=proxy_flag_cit,
        ),
        RevenueInstrument(
            name="Resource_Royalties",
            amounts=np.array([200.0, 210.0, 220.0, 230.0, 240.0]),
            alpha=alpha_royalties,
            use_proxy=False,    # based on revenue-base analysis
        ),
    ]

    # ------------------------------------------------------------------
    # Expenditure instruments (DCU millions)
    # ------------------------------------------------------------------
    # Wages/transfers: primarily nontradable (phi = 0.05, only minor tradable
    # component through imported goods in the consumption basket)
    # Capital expenditure: largely imported machinery; phi = 0.70
    exp_instruments = [
        ExpenditureInstrument(
            name="Wages_Transfers",
            amounts=np.array([350.0, 360.0, 371.0, 382.0, 394.0]),
            phi=0.05,
        ),
        ExpenditureInstrument(
            name="Capital_Expenditure",
            amounts=np.array([100.0, 105.0, 110.0, 115.0, 120.0]),
            phi=0.70,
        ),
    ]

    # ------------------------------------------------------------------
    # Debt block (DCU millions)
    # D_FX already expressed in domestic currency at baseline q0=1.
    # chi = 1: FX debt revalues 1-for-1 with the nominal exchange rate e.
    # psi_q_to_e = 1: short-horizon sticky-price mapping from q shock to e shock.
    # ------------------------------------------------------------------
    debt = DebtBlock(D_FX_0=600.0, D_LC=800.0, chi=1.0, psi_q_to_e=1.0)

    # ------------------------------------------------------------------
    # Financial assets: reserves (held in USD, so delta_A = 1 under chi=1)
    # ------------------------------------------------------------------
    fin_assets = FinancialAssets(A_fin_0=150.0, delta_A=1.0)

    # ------------------------------------------------------------------
    # Elasticity parameters
    # beta_T > beta_N: tradable revenues more sensitive to RER
    # gamma_T > gamma_N: tradable expenditures more sensitive to RER
    # ------------------------------------------------------------------
    elasticities = ElasticityParams(
        beta_T=np.full(T, 0.80),
        beta_N=np.full(T, 0.10),
        gamma_T=np.full(T, 0.60),
        gamma_N=np.full(T, 0.05),
    )

    # ------------------------------------------------------------------
    # Discount curve: flat at 6%
    # ------------------------------------------------------------------
    discount_curve = build_flat_discount_curve(0.06, T)

    # ------------------------------------------------------------------
    # Assemble parameters
    # ------------------------------------------------------------------
    params = SovereignSALMParams(
        revenue_instruments=rev_instruments,
        expenditure_instruments=exp_instruments,
        debt=debt,
        fin_assets=fin_assets,
        elasticities=elasticities,
        discount_curve=discount_curve,
        q0=q0,
        T=T,
    )

    # ------------------------------------------------------------------
    # BASELINE BALANCE SHEET (q = q0)
    # ------------------------------------------------------------------
    bs0 = compute_balance_sheet(params, q0)
    print("\nBASELINE BALANCE SHEET (q = q0 = 1.0)")
    print(bs0.summary().to_string())

    # ------------------------------------------------------------------
    # ELASTICITY DECOMPOSITION at baseline
    # ------------------------------------------------------------------
    er = compute_elasticity_analytical(params, bs0)
    print()
    print_elasticity_decomposition(er)

    # ------------------------------------------------------------------
    # SCENARIO ANALYSIS: q grid from 0.7 to 1.5 (depreciation shock)
    # ------------------------------------------------------------------
    q_grid = np.linspace(0.7, 1.5, 9).tolist()
    scenario_df = scenario_analysis(params, q_grid)
    print("\nSCENARIO ANALYSIS (selected columns)")
    print(
        scenario_df[["q", "A_total", "L_total", "leverage", "epsilon_numerical"]]
        .to_string(index=False, float_format="{:.4f}".format)
    )

    # ------------------------------------------------------------------
    # INTERPRETATION
    # ------------------------------------------------------------------
    print("\nINTERPRETATION")
    print(f"  Baseline leverage λ(q₀)       = {bs0.leverage:.4f}")
    dep_10pct = scenario_df[
        np.isclose(scenario_df["q"], 1.1, atol=0.05)
    ]["leverage"].iloc[0]
    print(f"  Leverage at q≈1.1 (+10% dep)  = {dep_10pct:.4f}")
    print(f"  Analytical ε_{{λ,q}}            = {er.epsilon_analytical:.4f}")
    print(
        f"  Interpretation: A 1% real depreciation changes leverage "
        f"by ~{er.epsilon_analytical:.2f}%."
    )
    if er.epsilon_analytical > 0:
        print("  RISK: Depreciation INCREASES leverage (liability-heavy).")
    else:
        print("  BUFFER: Depreciation REDUCES leverage (asset-heavy).")


# ============================================================================
# SECTION 9: ASSUMPTIONS CHECKLIST
# ============================================================================

ASSUMPTIONS_CHECKLIST = """
ASSUMPTIONS AND HOW TO RELAX THEM
===================================

SOURCE-BASED DEFINITIONS (non-negotiable)
------------------------------------------
[S1] RER definition: q = e * P_star / P  (IMF Finance & Development).
[S2] Sovereign economic balance sheet includes PV of future revenues and
     expenditures (IMF WP/12/241, IMF WP/19/290).
[S3] Revenue instrument classification follows IMF GFSM 2014, Chapter 5.
[S4] PV discounting: PV(X) = Σ X_t / Π(1+r_s) (IMF TNM/10/02).

MODELING ASSUMPTIONS (can be changed with caution)
----------------------------------------------------
[M1] FX debt is fundamentally revalued by the nominal exchange rate e.
     In q-space, its effective elasticity is eta_DFX,q = chi * psi_q_to_e,
     where chi = d ln(D_FX)/d ln(e) and psi_q_to_e = d ln(e)/d ln(q).
     Baseline: chi = 1 and psi_q_to_e = 1 under sticky prices.
     → Relax: set psi_q_to_e ≠ 1 if using a q shock that is not numerically
       identical to the nominal-FX shock relevant for debt valuation.

[M2] D_LC is invariant to q.
     → Relax: introduce FX-indexed domestic debt as a separate block with
       its own chi. Requires separate data on indexed instruments.

[M3] Discount rates are exogenous and invariant to q.
     → Relax: model sovereign risk premium as a function of lambda(q),
       creating a nonlinear feedback. Requires sovereign CDS data and
       estimation of risk-premium elasticity.

[M4] Power-law response functions X_t(q) = X_t0 * (q/q0)^eta.
     → Relax: use non-parametric or sector-estimated response functions.
       Requires time-series econometric estimation of revenue elasticities.

[M5] beta_T > beta_N (tradable revenues more RER-sensitive).
     → Verify empirically. Not enforced programmatically.

[M6] alpha_i via exports/GDP proxy (when approved).
     → Relax: use instrument-specific base analysis (e.g., customs duties
       are ~100% tradable; income tax from mining sector is high-alpha;
       VAT on domestically consumed services is low-alpha).

[M7] A_fin_0 has fixed delta_A.
     → Relax: decompose reserves (delta_A = 1 for USD reserves) from
       domestic equity stakes (delta_A ≈ 0) separately.

[M8] No contingent liabilities, no FX-indexed domestic debt, no derivatives.
     → Extend: add option-adjusted present values for guarantees; add
       separate CPI-indexed and FX-indexed domestic debt blocks.

[M9] Single bilateral q (not REER).
     → Relax: use currency-basket REER; requires mapping D_FX into
       currency-specific sub-blocks (USD, EUR, JPY, etc.) and computing
       a portfolio-weighted nominal pass-through chi together with an
       explicit q-to-e mapping psi_q_to_e.

IMPLEMENTATION DECISIONS
-------------------------
[I1] Numerical central-difference with h=1e-5.
     → For very large shocks (q/q0 >> 2), use arc-elasticity instead.

[I2] All amounts in domestic-currency units (no USD denomination internally).
     → The caller is responsible for converting D_FX_0 at the baseline e_0.

[I3] Exports/GDP proxy is only applied when caller explicitly approves it.
     → Always flag use_proxy=True on affected instruments.

[I4] Weights and elasticities are computed at q0 for the analytical formula.
     → For large shocks, use numerical elasticity which is evaluated
       at the shocked q.
"""


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)  # suppress proxy warnings in demo

    worked_example()
    print("\n" + "=" * 65)
    run_all_tests(verbose=True)
    print()
    print(ASSUMPTIONS_CHECKLIST)
