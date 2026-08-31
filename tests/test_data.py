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
    exclusion_report, scale_report, section_loads, section_nodes,
    station_diagram,
    station_drivers, configuration_report, transfer)

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


def block(title, *parts):
    """Formats a failure message so the numbers are visible in the report."""
    out = ["", "=" * 68, title, "=" * 68]
    out += [str(p) for p in parts if p is not None and str(p).strip()]
    return "\n".join(out)


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
def audit(solved, cg):
    """force_audit walks every card row, so it is computed once."""
    return force_audit(solved["df"], solved["pm"], cg, solved["sec"])


@pytest.fixture(scope="module")
def diag(solved):
    return station_diagram(solved["sec"])


CG_SOURCE = ""


@pytest.fixture(scope="module")
def cg(solved):
    """Vehicle CG. Override with SECTION_LOADS_CG="x,y,z"."""
    global CG_SOURCE
    env = os.environ.get("SECTION_LOADS_CG")
    if env:
        CG_SOURCE = "SECTION_LOADS_CG"
        return np.array([float(v) for v in env.split(",")])
    CG_SOURCE = ("mean of node coordinates (NOT the real CG -- set "
                 "SECTION_LOADS_CG)")
    return solved["df"][XCOLS].to_numpy(float).mean(axis=0)


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
        if moved.empty:
            continue
        worst = moved.assign(worst=moved.max(axis=1)).nlargest(8, "worst")
        where = g[g["node_id"].isin(worst.index)][
            ["node_id", "loadcase", "source_file"] + XCOLS]
        raise AssertionError(block(
            f"{cfg}: {len(moved)} node(s) have different coordinates "
            "between load cases",
            f"model extent {extent:,.4g}, tolerance {1e-6 * extent:,.4g}",
            f"worst movement {float(spread.to_numpy().max()):,.6g}",
            "",
            "worst nodes:",
            worst.to_string(),
            "",
            "their rows, by file:",
            where.sort_values(["node_id", "loadcase"]).head(24).to_string(
                index=False)))


def test_node_sets_match_after_declared_exclusions(loaded):
    """Every case in a configuration must hold the same nodes, once the ids
    declared in the exclude_nodes column of load_cases.csv are allowed for.

    The reference is the union of all nodes seen in that configuration. A case
    must equal that union minus its own declared exclusions -- so an undeclared
    absence fails, and a declared node that turns up anyway fails too.
    """
    _, _, load_cases, _, df = loaded
    declared = load_cases.set_index("loadcase")["exclude_nodes"].to_dict()

    for cfg, g in df.groupby("config_folder"):
        union = set(g["node_id"])
        files = g.drop_duplicates("loadcase").set_index("loadcase")["source_file"]
        problems = []
        for lc, gc in g.groupby("loadcase"):
            ids = set(gc["node_id"])
            drop = declared.get(lc, set())
            expected = union - drop
            undeclared = sorted(expected - ids)
            unexpected = sorted(ids & drop)
            if not undeclared and not unexpected:
                continue
            problems.append(
                f"  {lc}  {files.get(lc, '?')}"
                f"\n      {len(ids):,} nodes, {len(drop)} declared excluded"
                + (f"\n      MISSING but not declared ({len(undeclared)}): "
                   f"{undeclared[:12]}" if undeclared else "")
                + (f"\n      PRESENT but declared excluded ({len(unexpected)}): "
                   f"{unexpected[:12]}" if unexpected else ""))
        if not problems:
            continue
        raise AssertionError(block(
            f"{cfg}: node sets do not reconcile with the declared exclusions",
            f"union across the configuration: {len(union):,} nodes",
            "",
            "\n".join(problems),
            "",
            "Declare an expected absence in the exclude_nodes column of "
            "load_cases.csv, e.g. \"1000, 1001, 2000-2005\". Anything left "
            "here is a truncated export, a different model revision, or a "
            "region that produced no output."))


def test_declared_exclusions_match_something(loaded):
    """An id matching neither a card node nor a point mass does nothing, which
    is almost always a typo."""
    _, _, load_cases, point_masses, df = loaded
    rep = exclusion_report(df, point_masses, load_cases)
    if rep.empty:
        pytest.skip("no exclude_nodes column")
    bad = rep[rep["unknown"].map(len) > 0]
    if bad.empty:
        return
    raise AssertionError(block(
        "exclude_nodes ids match no card node and no point mass",
        bad[["loadcase", "declared", "unknown"]].to_string(index=False),
        "",
        f"card node ids run {df['node_id'].min()}..{df['node_id'].max()}, "
        f"point masses {sorted(point_masses['node_id'])[:8]}",
        "",
        rep.to_string(index=False)))


def test_excluded_point_masses_are_not_applied(solved):
    """The exclusion covers point mass ids, so their inertia must not appear."""
    pm_loads, load_cases = solved["pm"], solved["load_cases"]
    if "exclude_nodes" not in load_cases.columns:
        pytest.skip("no exclude_nodes column")
    bad = []
    for r in load_cases.itertuples(index=False):
        drop = r.exclude_nodes or set()
        applied = set(pm_loads.loc[pm_loads["loadcase"] == r.loadcase,
                                   "node_id"])
        leaked = sorted(applied & drop)
        if leaked:
            bad.append(f"  {r.loadcase}: {leaked}")
    assert not bad, block(
        "point masses declared excluded are still being applied",
        "\n".join(bad))


# --------------------------------------------------- nothing is lost ---------

def test_every_node_is_tagged(solved):
    rep = node_report(solved["df"])
    untagged = rep[rep["component"] == "UNTAGGED"]
    assert untagged.empty, f"\n{untagged.to_string(index=False)}"


def test_every_declared_component_has_nodes(solved):
    """A component with sections and a GID range but no nodes in the cards sums
    to zero, while whatever component does hold those nodes carries them. The
    name checks pass either way, so nothing else catches this.

    The vehicle is exempt: its sections span every node and it has no GID range
    of its own. Names are compared with norm(), as everywhere else.
    """
    rep = node_report(solved["df"]).copy()
    rep["key"] = rep["component"].map(norm)
    have = {k: v for k, v in zip(rep["key"], rep["in_sections"])}
    exempt = {norm(VEHICLE)} | {norm(i) for i in EXCLUDE_ITEMS}
    declared = {norm(c): c for c in solved["sections"]["component"]}
    missing = sorted(name for key, name in declared.items()
                     if key not in exempt and have.get(key, 0) == 0)
    if not missing:
        return
    raise AssertionError(block(
        "declared in sections.csv but no card nodes",
        f"components: {missing}",
        "",
        "Their sections sum to zero, and whatever component does hold those "
        "nodes carries them instead.",
        "",
        rep.drop(columns="key").to_string(index=False),
        "",
        f"exempt: {sorted(exempt)}   (vehicle plus EXCLUDE_ITEMS)"))


def test_vehicle_partition_sees_more_than_any_one_component(solved):
    """The vehicle sections bin every node, so unless a single component
    already holds them all, their node counts must exceed it. Identical counts
    mean the same nodes went into both partitions."""
    sec = solved["sec"]
    if not (sec["partition"] == "vehicle").any():
        pytest.skip("no vehicle partition")
    comps = set(sec.loc[sec["partition"] == "component", "component"].map(norm))
    if len(comps) < 2:
        pytest.skip("only one component, so the partitions coincide")
    lc = sec["loadcase"].iloc[0]
    g = sec[sec["loadcase"] == lc]
    veh = g.loc[g["partition"] == "vehicle", "n_nodes"].sum()
    per = g.loc[g["partition"] == "component"].groupby("component")["n_nodes"].sum()
    assert veh > per.max(), (
        f"vehicle partition holds {veh} node(s); largest component holds "
        f"{per.max()}. The vehicle bands are not picking up the other "
        f"components.\n{per.to_string()}")


def test_no_orphan_nodes_and_no_gaps(solved):
    rep = assignment_report(solved["bodies"], solved["sections"], VEHICLE)
    bad = rep[(rep["orphans"] > 0) | (rep["gaps"] > 0)]
    assert bad.empty, f"\n{bad.to_string(index=False)}"


def test_sections_account_for_every_card_row(audit):
    """Rows 1 to 3 of the audit must reconcile with the section totals."""
    for lc, g in audit.groupby("loadcase"):
        g = g.set_index("group")
        used = g.loc["3 card rows in sections", SIX].to_numpy(float)
        sect = g.loc["3a component sections", SIX].to_numpy(float)
        scale = max(np.abs(used).max(), 1.0)
        if np.allclose(used, sect, atol=1e-6 * scale):
            continue
        d = pd.DataFrame({"component": SIX, "card rows": used,
                          "sections": sect, "difference": sect - used})
        raise AssertionError(block(
            f"{lc}: sections do not equal the nodes they came from",
            d.to_string(index=False, float_format=lambda v: f"{v:,.4f}"),
            "",
            "Load is being lost or double counted between tagging and "
            "binning. Check orphans in assignment_report."))


def test_partitions_agree(solved, cg):
    r = check_partitions_agree(solved["sec"], cg)
    assert r["ok"], (f"worst {r['worst']:.4g} against scale {r['scale']:.4g}\n"
                     f"{r['err'].to_string(index=False)}")


def test_section_equals_its_own_nodes_about_the_cg(solved, cg):
    """Spot check: the centroid cancels when transferring to the CG.

    One load case and one body only -- section_nodes concatenates whichever
    bodies it is given, which is expensive on a large model.
    """
    sec, bodies = solved["sec"], solved["bodies"]
    cfgs = solved["load_cases"].set_index("loadcase")["config"]
    lc = sec["loadcase"].iloc[0]
    one = {cfgs[lc]: bodies[cfgs[lc]]}
    comp = sec[(sec["partition"] == "component") & (sec["loadcase"] == lc)]
    for r in comp.drop_duplicates("section_id").head(8).itertuples(index=False):
        n = section_nodes(one, r.loadcase, r.section_id)
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
    """A stray 386.09 or a sign error shows up here.

    Against the PER-CASE mass total, since a case may exclude a point mass.
    """
    pm, cases = solved["pm"], solved["load_cases"]
    sheet = solved["point_masses"]
    for r in cases.itertuples(index=False):
        drop = getattr(r, "exclude_nodes", None) or set()
        total = sheet.loc[~sheet["node_id"].isin(drop), "mass"].sum()
        for col, n in zip(FCOLS, (r.nx, r.ny, r.nz)):
            got = pm.loc[pm["loadcase"] == r.loadcase, col].sum()
            want = INERTIA_SIGN * total * n / ACCEL_TO_G
            assert got == pytest.approx(want, rel=1e-9, abs=1e-9), (
                f"{r.loadcase} {col}: {got:,.6g} != {want:,.6g} "
                f"(mass {total:,.4f} after {len(drop)} exclusion(s))")


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


UNIT_FACTORS = {"gram": 1 / 453.592, "kg": 2.20462, "slug": 32.174,
                "lbf-s^2/in": 386.09}

# Point mass total absolute load over card total absolute load. Secondary
# structure should be a fraction of the vehicle, so 1.0 is already generous.
# Raise it if your point masses are genuinely a large share.
PM_RATIO_LIMIT = 1.0


def test_point_mass_load_is_reported_against_the_card_load(solved, audit):
    """Units sanity. F = -m*N is self-consistent in ANY mass unit, so the only
    signal is the size of the result next to the card loads."""
    pm_sheet, df = solved["point_masses"], solved["df"]
    # Scale against the total ABSOLUTE nodal force, not the resultant. A
    # balanced free-flight case has a resultant near zero, which would inflate
    # any ratio taken against it.
    scale = df.groupby("loadcase")[FCOLS].apply(
        lambda g: float(np.abs(g.to_numpy(float)).sum()))
    ratios = {}
    for lc, g in audit.groupby("loadcase"):
        pmass = float(np.abs(g.set_index("group").loc[
            "4 point mass loads", ["Fx", "Fy", "Fz"]].to_numpy(float)).sum())
        if scale.get(lc):
            ratios[lc] = pmass / scale[lc]
    worst_lc = max(ratios, key=ratios.get, default=None)
    worst = ratios.get(worst_lc, 0.0)
    # Totals of absolute components, so node count is not a factor: 100k small
    # nodal forces sum large, 10 point masses sum to their own total. A
    # max-to-max comparison would be meaningless here.
    if worst < PM_RATIO_LIMIT:
        return

    total = float(pm_sheet["mass"].sum())
    nmax = float(solved["load_cases"][["nx", "ny", "nz"]].abs().to_numpy().max())
    cards = float(scale[worst_lc])
    table = "\n".join(
        f"    {name:14s} mass {total * f:14,.2f}   load/card "
        f"{abs(total * f * nmax) / cards:10,.4f}"
        for name, f in UNIT_FACTORS.items())
    raise AssertionError(block(
        "point mass loads are out of scale with the card loads",
        f"worst case {worst_lc}: point mass load is {worst:,.2f}x the total "
        f"absolute card load",
        "",
        f"  mass column total   {total:,.4f}",
        f"  mass min / max      {pm_sheet['mass'].min():,.4f} / "
        f"{pm_sheet['mass'].max():,.4f}   over {len(pm_sheet)} row(s)",
        f"  largest |n|         {nmax:,.4f}",
        f"  total |F| in cards  {cards:,.2f}   (sum of |fx|+|fy|+|fz|)",
        "",
        "  ratio per case: " + str({k: round(v, 3) for k, v in ratios.items()}),
        "",
        "  if the column were in these units instead:",
        table,
        "",
        "  F = mass * n must give force directly, so the column must be lbm.",
        "  Loads too LARGE rules out slug, snail and kg -- those make F too",
        "  small. Check the first rows of point_masses.csv:",
        pm_sheet.head(5).to_string(index=False)))


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
    """|M| / |F| implies a lever arm; far outside the model is suspicious.

    Section moments are about their OWN centroids, so a large arm means the
    centroid is far from the load it represents.
    """
    extent = float(np.ptp(solved["df"][XCOLS].to_numpy(float), axis=0).max())
    sec = solved["sec"].copy()
    F = np.linalg.norm(sec[["Fx", "Fy", "Fz"]].to_numpy(float), axis=1)
    M = np.linalg.norm(sec[["Mx", "My", "Mz"]].to_numpy(float), axis=1)
    sec["arm"] = np.where(F > 1e-9 * max(F.max(), 1.0), M / np.maximum(F, 1e-30),
                          0.0)
    bad = sec[sec["arm"] > 5.0 * extent]
    if bad.empty:
        return
    cols = ["loadcase", "partition", "component", "section_id", "n_nodes",
            "cen_x", "cen_y", "cen_z", "arm"]
    raise AssertionError(block(
        "implied moment arms exceed the model",
        f"model extent {extent:,.3f}, threshold {5 * extent:,.3f}",
        f"{len(bad)} of {len(sec)} section row(s) exceed it",
        "",
        bad.nlargest(10, "arm")[cols].to_string(index=False),
        "",
        f"CG used: {np.round(cg, 4).tolist()}  from {CG_SOURCE}"))


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

def test_diagram_last_cut_equals_the_component_total(solved, diag):
    """The running sum must close on the component it came from."""
    sec = solved["sec"]
    for (lc, cfg, comp), g in diag.groupby(["loadcase", "configuration",
                                            "component"]):
        last = g.loc[g["sections"].idxmax()]
        want = sec[(sec["loadcase"] == lc) & (sec["component"] == comp)
                   & (sec["configuration"] == cfg)][["Fx", "Fy", "Fz"]
                                                    ].to_numpy(float).sum(axis=0)
        got = np.array([last["Fx"], last["Fy"], last["Fz"]])
        scale = max(np.abs(want).max(), 1.0)
        assert np.allclose(got, want, atol=1e-6 * scale), (lc, comp)


def test_every_load_case_appears_in_the_diagram(solved, diag):
    missing = set(solved["load_cases"]["loadcase"]) - set(diag["loadcase"])
    assert not missing, f"missing from the diagram: {sorted(missing)}"


def test_drivers_cover_every_component(diag):
    drivers = station_drivers(diag)
    assert set(drivers["component"]) == set(diag["component"])