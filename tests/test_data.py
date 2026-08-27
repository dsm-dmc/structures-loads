"""Checks against real data files.

Skipped when no data directory is present, so a clean checkout still passes.
Point it elsewhere with an environment variable:

    SECTION_LOADS_DATA=/path/to/data poetry run pytest tests/test_data.py -q

Set the vehicle and exclude names to match your tables if they differ.

These assert engineering properties, not code behaviour: that every card row is
accounted for, that the two partitions agree, that symmetric components mirror,
that the units come out right. A failure here means the data or the tables are
wrong, not the code.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from section_loads import (
    FCOLS, SIX, XCOLS, assignment_report, check_card_configurations,
    check_component_coverage, check_configurations, check_partitions_agree,
    check_sections_per_configuration, force_audit, node_report, norm,
    point_mass_loads, read_force_cards, read_tables, run_sections,
    section_loads, section_nodes, station_diagram, station_drivers, transfer)

VEHICLE = "Vehicle"
EXCLUDE_ITEMS = {"Point Masses"}
INERTIA_SIGN = -1.0
ACCEL_TO_G = 1.0

# Left/right pairs that should mirror in a symmetric case, and the load case
# names that are symmetric. Leave SYMMETRIC_CASES empty to skip that check.
MIRROR_PAIRS = [("LH Wing", "RH Wing")]
SYMMETRIC_CASES: list[str] = []

DATA = Path(os.environ.get("SECTION_LOADS_DATA",
                           Path(__file__).resolve().parents[1] / "data"))
pytestmark = pytest.mark.skipif(
    not (DATA / "sections.csv").exists(),
    reason=f"no data at {DATA}; set SECTION_LOADS_DATA to run these")


# ------------------------------------------------------------- fixtures ------

@pytest.fixture(scope="module")
def tables():
    return read_tables(DATA)


@pytest.fixture(scope="module")
def loaded(tables):
    sections, node_ranges, load_cases, point_masses = tables
    df, load_cases = read_force_cards(DATA, load_cases)
    return sections, node_ranges, load_cases, point_masses, df


@pytest.fixture(scope="module")
def solved(loaded):
    sections, node_ranges, load_cases, point_masses, df = loaded
    from section_loads import tag_components
    df = tag_components(df, node_ranges, EXCLUDE_ITEMS)
    pm = point_mass_loads(point_masses, load_cases, INERTIA_SIGN, ACCEL_TO_G)
    sec, bodies = run_sections(df, sections, load_cases, VEHICLE)
    return dict(sections=sections, node_ranges=node_ranges,
                load_cases=load_cases, point_masses=point_masses, df=df,
                pm=pm, sec=sec, bodies=bodies)


@pytest.fixture(scope="module")
def cg(solved):
    """Vehicle CG. Override with SECTION_LOADS_CG="x,y,z"."""
    env = os.environ.get("SECTION_LOADS_CG")
    if env:
        return np.array([float(v) for v in env.split(",")])
    X = solved["df"][XCOLS].to_numpy(float)
    return X.mean(axis=0)


# --------------------------------------------------- tables are consistent ---

def test_tables_load(tables):
    sections, node_ranges, load_cases, point_masses = tables
    assert len(sections) and len(node_ranges) and len(load_cases)


def test_component_names_match_both_ways(tables):
    sections, node_ranges, _, _ = tables
    check_component_coverage(sections, node_ranges, VEHICLE, EXCLUDE_ITEMS)


def test_configurations_are_served(tables):
    sections, _, load_cases, _ = tables
    r = check_configurations(sections, load_cases)
    assert not r["unserved"], f"no sections for {sorted(r['unserved'])}"


def test_every_item_has_sections_in_every_configuration(tables):
    sections, node_ranges, load_cases, _ = tables
    gaps = check_sections_per_configuration(sections, node_ranges, load_cases,
                                            VEHICLE, EXCLUDE_ITEMS)
    assert gaps.empty, f"\n{gaps.to_string(index=False)}"


def test_card_folders_agree_with_the_sheet(loaded):
    _, _, load_cases, _, df = loaded
    out = check_card_configurations(df, load_cases)
    assert out["agree"].all(), f"\n{out[~out['agree']].to_string(index=False)}"


def test_geometry_is_identical_within_a_configuration(loaded):
    """Geometry may differ BETWEEN configurations -- a deployed fin moves, the
    model changes. Within one configuration a node must not move, since the
    same grid is being loaded by different cases."""
    _, _, _, _, df = loaded
    extent = float(np.ptp(df[XCOLS].to_numpy(float), axis=0).max())
    for cfg, g in df.groupby("config_folder"):
        agg = g.groupby("node_id")[XCOLS].agg(["min", "max"])
        spread = pd.DataFrame(
            {c: agg[(c, "max")] - agg[(c, "min")] for c in XCOLS})
        moved = spread[(spread > 1e-6 * extent).any(axis=1)]
        assert moved.empty, (
            f"{cfg}: {len(moved)} node(s) move between cases, worst "
            f"{spread.to_numpy().max():.4g}\n"
            f"{moved.head().to_string()}")


def test_node_sets_are_identical_within_a_configuration(loaded):
    """Enveloping across cases with different node sets is not meaningful."""
    _, _, _, _, df = loaded
    for cfg, g in df.groupby("config_folder"):
        sizes = g.groupby("loadcase")["node_id"].nunique()
        assert sizes.nunique() == 1, f"{cfg}:\n{sizes.to_string()}"


# --------------------------------------------------- nothing is lost ---------

def test_every_node_is_tagged(solved):
    rep = node_report(solved["df"])
    untagged = rep[rep["component"] == "UNTAGGED"]
    assert untagged.empty, f"\n{untagged.to_string(index=False)}"


def test_no_orphan_nodes_and_no_gaps(solved):
    rep = assignment_report(solved["bodies"], solved["sections"], VEHICLE)
    bad = rep[(rep["orphans"] > 0) | (rep["gaps"] > 0)]
    assert bad.empty, f"\n{bad.to_string(index=False)}"


def test_sections_account_for_every_card_row(solved, cg):
    """Rows 1 to 3 of the audit must reconcile with the section totals."""
    au = force_audit(solved["df"], solved["pm"], cg, solved["sec"])
    for lc, g in au.groupby("loadcase"):
        g = g.set_index("group")
        used = g.loc["3 card rows in sections", SIX].to_numpy(float)
        sect = g.loc["3a component sections", SIX].to_numpy(float)
        scale = max(np.abs(used).max(), 1.0)
        assert np.allclose(used, sect, atol=1e-6 * scale), (
            f"{lc}: sections do not equal the nodes they came from")


def test_partitions_agree(solved, cg):
    r = check_partitions_agree(solved["sec"], cg)
    assert r["ok"], (f"worst {r['worst']:.4g} against scale {r['scale']:.4g}\n"
                     f"{r['err'].to_string(index=False)}")


def test_section_equals_its_own_nodes_about_the_cg(solved, cg):
    """Spot check: the centroid cancels when transferring to the CG."""
    sec, bodies = solved["sec"], solved["bodies"]
    comp = sec[sec["partition"] == "component"]
    for r in comp.drop_duplicates("section_id").head(12).itertuples(index=False):
        n = section_nodes(bodies, r.loadcase, r.section_id)
        if n.empty:
            continue
        F = n[FCOLS].to_numpy(float)
        got = np.concatenate([F.sum(axis=0),
                              np.cross(n[XCOLS].to_numpy(float) - cg, F)
                              .sum(axis=0)])
        Fs = np.array([r.Fx, r.Fy, r.Fz])
        Ms = np.array([r.Mx, r.My, r.Mz])
        _, Mcg = transfer(Fs, Ms, [r.cen_x, r.cen_y, r.cen_z], cg)
        want = np.concatenate([Fs, Mcg])
        scale = max(np.abs(want).max(), 1.0)
        assert np.allclose(got, want, atol=1e-6 * scale), r.section_id


# --------------------------------------------------- units and magnitudes ----

def test_point_mass_force_equals_mass_times_load_factor(solved):
    """A stray 386.09 or a sign error shows up here."""
    pm, cases = solved["pm"], solved["load_cases"]
    total = solved["point_masses"]["mass"].sum()
    for r in cases.itertuples(index=False):
        got = pm.loc[pm["loadcase"] == r.loadcase, "fz"].sum()
        want = INERTIA_SIGN * total * r.nz / ACCEL_TO_G
        assert got == pytest.approx(want, rel=1e-9, abs=1e-9), r.loadcase


def test_point_mass_total_matches_the_mass_statement(solved):
    """A units error cannot be found internally -- it needs an outside number.

    Set SECTION_LOADS_PM_MASS to the point mass total from the mass properties
    statement. Without it this is skipped, because a mass scaled by 386 still
    satisfies every internal relation.
    """
    expected = os.environ.get("SECTION_LOADS_PM_MASS")
    if not expected:
        pytest.skip("set SECTION_LOADS_PM_MASS to the mass statement total")
    got = float(solved["point_masses"]["mass"].sum())
    assert got == pytest.approx(float(expected), rel=0.01), (
        f"point_masses.csv totals {got:,.2f}; statement says {expected}")


def test_point_mass_load_is_reported_against_the_card_load(solved, cg):
    """Not a pass/fail on magnitude -- prints the ratio so a units error is
    visible. Only an absurd ratio fails."""
    au = force_audit(solved["df"], solved["pm"], cg)
    ratios = {}
    for lc, g in au.groupby("loadcase"):
        g = g.set_index("group")
        cards = np.abs(g.loc["1 all card rows", ["Fx", "Fy", "Fz"]]
                       .to_numpy(float)).max()
        pmass = np.abs(g.loc["4 point mass loads", ["Fx", "Fy", "Fz"]]
                       .to_numpy(float)).max()
        if cards:
            ratios[lc] = pmass / cards
    print("\npoint mass load / card load:",
          {k: round(v, 4) for k, v in ratios.items()})
    worst = max(ratios.values(), default=0.0)
    assert worst < 100.0, (
        f"point mass load is {worst:.0f}x the card load -- likely a mass unit "
        "mismatch (lbm vs slug, or a stray 386.09)")


def test_no_nan_or_inf_anywhere(solved):
    for name in ("sec", "pm"):
        f = solved[name]
        num = f.select_dtypes(include=[np.number])
        assert np.isfinite(num.to_numpy()).all(), f"non-finite values in {name}"


def test_no_section_is_empty(solved):
    """A section with no nodes is a bracketing error, not a light bay."""
    sec = solved["sec"]
    empty = sec[(sec["partition"] == "component") & (sec["n_nodes"] == 0)]
    assert empty.empty, (
        f"\n{empty[['loadcase', 'component', 'section_id']].to_string(index=False)}")


def test_moment_arms_are_within_the_model(solved, cg):
    """|M| / |F| implies a lever arm; far outside the model means a bad CG."""
    X = solved["df"][XCOLS].to_numpy(float)
    extent = float(np.ptp(X, axis=0).max())
    sec = solved["sec"]
    F = np.linalg.norm(sec[["Fx", "Fy", "Fz"]].to_numpy(float), axis=1)
    M = np.linalg.norm(sec[["Mx", "My", "Mz"]].to_numpy(float), axis=1)
    ok = F > 1e-9 * max(F.max(), 1.0)
    arm = M[ok] / F[ok]
    assert arm.max() < 5.0 * extent, (
        f"implied arm {arm.max():.3g} against model extent {extent:.3g}")


# --------------------------------------------------- symmetry ---------------

@pytest.mark.parametrize("left,right", MIRROR_PAIRS)
def test_mirror_components_mirror_in_symmetric_cases(solved, left, right):
    if not SYMMETRIC_CASES:
        pytest.skip("set SYMMETRIC_CASES to enable")
    sec = solved["sec"]
    for lc in SYMMETRIC_CASES:
        g = sec[(sec["loadcase"] == lc) & (sec["partition"] == "component")]
        a = g[g["component"] == left][SIX].to_numpy(float).sum(axis=0)
        b = g[g["component"] == right][SIX].to_numpy(float).sum(axis=0)
        # Fx, Fz, My mirror equal; Fy, Mx, Mz mirror negated
        expect = np.array([b[0], -b[1], b[2], -b[3], b[4], -b[5]])
        scale = max(np.abs(a).max(), np.abs(b).max(), 1.0)
        assert np.allclose(a, expect, atol=1e-6 * scale), f"{lc} {left}/{right}"


# --------------------------------------------------- diagrams ---------------

def test_diagram_last_cut_equals_the_component_total(solved):
    """The running sum must close on the component it came from."""
    sec = solved["sec"]
    diag = station_diagram(sec)
    for (lc, cfg, comp), g in diag.groupby(["loadcase", "configuration",
                                            "component"]):
        last = g.loc[g["sections"].idxmax()]
        want = sec[(sec["loadcase"] == lc) & (sec["component"] == comp)
                   & (sec["configuration"] == cfg)][["Fx", "Fy", "Fz"]
                                                    ].to_numpy(float).sum(axis=0)
        got = np.array([last["Fx"], last["Fy"], last["Fz"]])
        scale = max(np.abs(want).max(), 1.0)
        assert np.allclose(got, want, atol=1e-6 * scale), (lc, comp)


def test_every_load_case_appears_in_the_diagram(solved):
    diag = station_diagram(solved["sec"])
    missing = set(solved["load_cases"]["loadcase"]) - set(diag["loadcase"])
    assert not missing, f"missing from the diagram: {sorted(missing)}"


def test_drivers_cover_every_component(solved):
    diag = station_diagram(solved["sec"])
    drivers = station_drivers(diag)
    assert set(drivers["component"]) == set(diag["component"])