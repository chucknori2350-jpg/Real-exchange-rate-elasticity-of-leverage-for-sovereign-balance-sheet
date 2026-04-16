from reer_leverage_elasticity import (
    BalanceSheetSnapshot,
    ExposureComponent,
    LinearOneForOneRevaluationBalanceSheet,
    SideBreakdown,
    build_standard_snapshot_from_current_values,
)


def test_linear_model_matches_snapshot_at_current_point() -> None:
    model = LinearOneForOneRevaluationBalanceSheet(
        liabilities_domestic_fixed=60.0,
        liabilities_fx_notional=20.0,
        assets_domestic_fixed=120.0,
        assets_fx_notional=30.0,
    )
    q = 1.5
    snap = build_standard_snapshot_from_current_values(
        current_liabilities_domestic_fixed=60.0,
        current_liabilities_fx_sensitive=q * 20.0,
        current_assets_domestic_fixed=120.0,
        current_assets_fx_sensitive=q * 30.0,
    )
    assert abs(model.leverage(q) - snap.leverage) < 1e-12
    assert abs(model.leverage_real_fx_elasticity(q) - snap.leverage_real_fx_elasticity) < 1e-12


def test_local_elasticity_matches_small_finite_difference() -> None:
    model = LinearOneForOneRevaluationBalanceSheet(
        liabilities_domestic_fixed=60.0,
        liabilities_fx_notional=20.0,
        assets_domestic_fixed=120.0,
        assets_fx_notional=30.0,
    )
    q0 = 1.5
    q1 = q0 * 1.000001
    approx = model.finite_move_elasticity(q0, q1)
    exact_local = model.leverage_real_fx_elasticity(q0)
    assert abs(approx - exact_local) < 1e-6


def test_generic_snapshot_supports_partial_indexation() -> None:
    assets = SideBreakdown.from_iterable(
        [
            ExposureComponent("domestic_assets", 100.0, 0.0),
            ExposureComponent("export_receivables_partial", 50.0, 0.6),
        ]
    )
    liabilities = SideBreakdown.from_iterable(
        [
            ExposureComponent("local_debt", 70.0, 0.0),
            ExposureComponent("fx_debt", 30.0, 1.0),
            ExposureComponent("fx_linked_local_bond", 20.0, 0.8),
        ]
    )
    snap = BalanceSheetSnapshot(assets=assets, liabilities=liabilities)
    assert abs(snap.asset_real_fx_elasticity - 0.2) < 1e-12
    assert abs(snap.liability_real_fx_elasticity - ((30.0 / 120.0) + (20.0 / 120.0) * 0.8)) < 1e-12
    assert abs(snap.leverage_real_fx_elasticity - (0.3833333333333333 - 0.2)) < 1e-12
