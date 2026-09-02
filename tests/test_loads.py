"""Unit tests. No data files -- these pass or fail on the code alone.

Layers, in order:
    1  parsing and naming
    2  input file validation
    3  binning
    4  configuration resolution
    5  coverage
    6  moments and partitions
    7  reporting and export
    8  station diagrams
"""
import json
import numpy as np
import pandas as pd
import pytest

from section_loads import (
    FCOLS, SIX, XCOLS, assign_sections, bracket, canon_headers, cg_breakdown,
    check_component_coverage, check_configurations, check_partitions_agree,
    check_sections_per_configuration, config_from_path, effective_sections,
    export_point_mass_loads, export_section_loads, force_audit,
    loadcase_from_stem, norm, norm_loadcase, parse_node_list, parse_sta_dir,
    point_mass_loads, exclusion_report, apply_exclusions,
    read_force_cards, read_tables, run_sections, section_loads, section_nodes,
    station_of, tag_components, transfer,
    station_diagram, station_drivers, station_axes, station_columns,
    plot_all_station_diagrams, panel_drivers, plot_station_diagram, ROLES,
    plot_station_diagram)

CG = np.array([50.0, 0.0, 0.0])


LAT = {"x": "y", "y": "x", "z": "x"}


def sections_frame(rows):
    """rows: (component, config, label, section_id, dir, lo, hi, cx, cy, cz)"""
    df = pd.DataFrame(rows, columns=[
        "component", "configuration", "label", "section_id", "sta_dir",
        "sta_start", "sta_end", "cen_x", "cen_y", "cen_z"])
    df["lat_dir"] = df["sta_dir"].map(LAT)
    df["comp_key"] = df["component"].map(norm)
    df["config"] = df["configuration"].str.upper()
    lo, hi = zip(*[bracket(a, b) for a, b in zip(df.sta_start, df.sta_end)])
    df["sta_lo"], df["sta_hi"] = lo, hi
    return df


def ranges_frame(items):
    return pd.DataFrame({
        "gid_start": [1000 * (i + 1) for i in range(len(items))],
        "gid_end": [1000 * (i + 1) + 999 for i in range(len(items))],
        "item": items, "comp_key": [norm(i) for i in items]})


def nodes_frame(xs, component="Fuselage", loadcase="LC1", fz=10.0):
    return pd.DataFrame({
        "node_id": range(1000, 1000 + len(xs)), "x": list(xs), "y": 0.0,
        "z": 0.0, "fx": 0.0, "fy": 0.0, "fz": fz, "loadcase": loadcase,
        "component": component, "comp_key": norm(component), "excluded": False})


def cases_frame(pairs):
    """pairs: (loadcase, configuration)"""
    return pd.DataFrame({
        "loadcase": [a for a, _ in pairs],
        "configuration": [b for _, b in pairs],
        "config": [b for _, b in pairs], "nx": 0.0, "ny": 0.0, "nz": 1.0})


TWO_BAYS = sections_frame([
    ("Fuselage", "ALL", "fwd", "F1", "x", 0.0, 20.0, 10.0, 0.0, 0.0),
    ("Fuselage", "ALL", "aft", "F2", "x", 20.0, 40.0, 30.0, 0.0, 0.0)])


# --------------------------------------------------------------- parsing ------

@pytest.mark.parametrize("value,expect", [("x", "x"), ("Z", "z"), (" y ", "y")])
def test_parse_sta_dir(value, expect):
    assert parse_sta_dir(value) == expect


@pytest.mark.parametrize("bad", ["span", "-y", ""])
def test_bad_sta_dir_raises(bad):
    with pytest.raises(ValueError):
        parse_sta_dir(bad)


def test_station_is_the_raw_coordinate():
    frame = pd.DataFrame({"x": [0.0], "y": [-38.0], "z": [0.0]})
    assert station_of(frame, "y")[0] == pytest.approx(-38.0)


@pytest.mark.parametrize("a,b", [(-38.0, -18.0), (-18.0, -38.0)])
def test_brackets_take_either_order(a, b):
    lo, hi = bracket(a, b)
    assert lo < hi and {lo, hi} == {a, b}


def test_headers_ignore_case_and_spacing():
    raw = pd.DataFrame(columns=["GID Start", "gid end", " Item "])
    assert list(canon_headers(raw).columns) == ["gid_start", "gid_end", "item"]


def test_blank_rows_are_dropped():
    raw = pd.DataFrame({"item": ["Fuselage", None], "gid_start": [1000, None]})
    assert len(canon_headers(raw)) == 1


@pytest.mark.parametrize("raw", [101, "101", "LC101", "lc101", 101.0])
def test_loadcase_keys_normalise(raw):
    """The sheet may hold 101 while filenames carry LC101."""
    assert norm_loadcase(raw) == "LC101"


def test_contiguous_digits_split_on_the_known_list():
    """LC10110002 is LC101 + 10002; only the known list finds the boundary."""
    assert loadcase_from_stem("run_LC10110002_x", ["LC101", 102]) == "LC101"


def test_longest_known_loadcase_wins():
    assert loadcase_from_stem("x_LC1010002_y", ["LC10", "LC101"]) == "LC101"


@pytest.mark.parametrize("folder,expect", [
    ("CC_Force_Cards", "CC"), ("cc", "CC"), ("Force_Cards_FF", "FF")])
def test_configuration_found_in_a_longer_folder_name(tmp_path, folder, expect):
    d = tmp_path / folder
    d.mkdir()
    f = d / "run_LC101_x.txt"
    f.write_text("x")
    assert config_from_path(f, tmp_path, ["CC", "FF"]) == expect


# ------------------------------------------------------------- load cases -----

def cards(tmp_path, stems):
    for sub, stem in stems:
        d = tmp_path / sub
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{stem}.txt").write_text("$h\n$h\nhdr\n1000,0,0,0,0,0,1\n")
    return tmp_path


THREE = [("CC_Force_Cards", "run_LC10110002_x"),
         ("CC_Force_Cards", "run_LC10210008_x"),
         ("FF_Force_Cards", "run_LC20100245_x")]
THREE_CASES = cases_frame([("LC101", "CC"), ("LC102", "CC"), ("LC201", "FF")])


def test_filter_narrows_cards_and_cases_together(tmp_path):
    df, lcs = read_force_cards(cards(tmp_path, THREE), THREE_CASES,
                               only=["LC101", 201])
    assert set(df["loadcase"]) == {"LC101", "LC201"}
    assert sorted(lcs["loadcase"]) == ["LC101", "LC201"]


def test_unknown_loadcase_in_the_filter_raises(tmp_path):
    with pytest.raises(ValueError):
        read_force_cards(cards(tmp_path, THREE), THREE_CASES, only=["LC999"])


def test_two_files_for_one_case_refuse_to_merge(tmp_path):
    root = cards(tmp_path, [("CC_Force_Cards", "a_LC101_x"),
                            ("CC_Force_Cards", "b_LC101_y")])
    with pytest.raises(ValueError, match="matched by two files"):
        read_force_cards(root, THREE_CASES)


def test_configuration_comes_from_the_folder(tmp_path):
    df, _ = read_force_cards(cards(tmp_path, THREE), THREE_CASES)
    got = df.drop_duplicates("loadcase").set_index("loadcase")["config_folder"]
    assert got["LC101"] == "CC" and got["LC201"] == "FF"


# ----------------------------------------------------------- point masses -----

def test_one_lbm_at_one_g_is_one_lbf():
    """A stray 386.09 in the inertia path shows up here."""
    pm = pd.DataFrame({"node_id": [1], "mass": [1.0], "x": [0.0], "y": [0.0],
                       "z": [0.0]})
    out = point_mass_loads(pm, cases_frame([("LC1", "CC")]))
    assert out["fz"].iloc[0] == pytest.approx(-1.0)


def test_missing_point_mass_coordinates_raise():
    pm = pd.DataFrame({"node_id": [1], "mass": [1.0]})
    with pytest.raises(KeyError):
        point_mass_loads(pm, cases_frame([("LC1", "CC")]))


# ---------------------------------------------------------------- binning -----

def test_shared_boundary_node_goes_to_the_lower_section():
    """(lo, hi] bins, matching the reference SUMIFS."""
    assert assign_sections(nodes_frame([20.0]), TWO_BAYS,
                           "CC")["section_id"].iloc[0] == "F1"


def test_lowest_section_is_closed_at_the_bottom():
    assert assign_sections(nodes_frame([0.0]), TWO_BAYS,
                           "CC")["section_id"].iloc[0] == "F1"


def test_every_node_plane_lands_in_one_bay():
    body = assign_sections(nodes_frame([0.0, 10.0, 20.0, 30.0, 40.0]),
                           TWO_BAYS, "CC")
    assert body["section_id"].tolist() == ["F1", "F1", "F1", "F2", "F2"]


def test_overlapping_sections_raise():
    dup = pd.concat([TWO_BAYS, TWO_BAYS.iloc[[0]].assign(section_id="DUP")],
                    ignore_index=True)
    with pytest.raises(ValueError, match="overlaps"):
        assign_sections(nodes_frame([10.0]), dup, "CC")


def test_excluded_nodes_are_held_out():
    df = nodes_frame([10.0, 30.0])
    df.loc[1, "excluded"] = True
    assert len(assign_sections(df, TWO_BAYS, "CC")) == 1


def test_untagged_nodes_get_no_component():
    df = pd.DataFrame({"node_id": [1500, 7777], "x": 0.0, "y": 0.0, "z": 0.0,
                       "fx": 0.0, "fy": 0.0, "fz": 1.0, "loadcase": "LC1"})
    out = tag_components(df, ranges_frame(["Fuselage"]))
    assert list(out["component"]) == ["Fuselage", "UNTAGGED"]


# ------------------------------------------------------------ configuration ---

MIXED = sections_frame([
    ("Fuselage", "CC", "nose", "F1_CC", "x", 0.0, 20.0, 10.0, 0.0, 0.0),
    ("Fuselage", "FF", "nose", "F1_FF", "x", 0.0, 20.0, 10.0, 0.0, 0.0),
    ("RH Wing", "ALL", "rw", "RW1", "y", 18.0, 38.0, 62.0, 28.0, 0.5)])


def test_all_sections_apply_to_every_configuration():
    for cfg in ("FF", "CC"):
        assert "RW1" in set(effective_sections(MIXED, cfg)["section_id"])


def test_configuration_sections_do_not_leak():
    assert set(effective_sections(MIXED, "FF")["section_id"]) == {"F1_FF", "RW1"}
    assert set(effective_sections(MIXED, "CC")["section_id"]) == {"F1_CC", "RW1"}


def test_component_mixing_all_and_configuration_is_rejected(tmp_path):
    sec = sections_frame([
        ("Fuselage", "ALL", "a", "F_ALL", "x", 0.0, 20.0, 10.0, 0.0, 0.0),
        ("Fuselage", "CC", "b", "F_CC", "x", 0.0, 20.0, 10.0, 0.0, 0.0)])
    write_tables(tmp_path, sec)
    with pytest.raises(ValueError, match="mixing ALL"):
        read_tables(tmp_path)


def test_loadcase_configuration_all_is_rejected():
    with pytest.raises(ValueError, match="not meaningful"):
        check_configurations(TWO_BAYS, cases_frame([("LC1", "ALL")]))


def test_loadcase_resolving_no_sections_raises():
    """Zero rows for a case makes every later check pass vacuously."""
    sec = sections_frame([
        ("Fuselage", "CC", "a", "F1", "x", 0.0, 40.0, 20.0, 0.0, 0.0)])
    df = pd.concat([nodes_frame([10.0], loadcase="A"),
                    nodes_frame([10.0], loadcase="B")], ignore_index=True)
    with pytest.raises(ValueError, match="resolved no sections"):
        run_sections(df, sec, cases_frame([("A", "CC"), ("B", "FF")]))


def write_tables(d, sections):
    sections.drop(columns=["comp_key", "config", "sta_lo", "sta_hi"],
                  errors="ignore").to_csv(d / "sections.csv", index=False)
    pd.DataFrame({"gid_start": [1000], "gid_end": [1999],
                  "item": [sections["component"].iloc[0]]}).to_csv(
        d / "node_ranges.csv", index=False)
    cases_frame([("LC1", "CC")]).drop(columns="config").to_csv(
        d / "load_cases.csv", index=False)
    pd.DataFrame({"node_id": [9001], "mass": [1.0], "x": [0.0], "y": [0.0],
                  "z": [0.0]}).to_csv(d / "point_masses.csv", index=False)


def test_same_section_id_in_two_configurations_is_allowed(tmp_path):
    """A section_id names a bay, so F2 exists once per configuration."""
    write_tables(tmp_path, MIXED.assign(section_id="F1"))
    assert len(read_tables(tmp_path)[0]) == 3


# -------------------------------------------------------------- coverage ------

def test_coverage_matches_both_ways():
    check_component_coverage(TWO_BAYS, ranges_frame(["Fuselage"]))


def test_section_component_with_no_gid_range_raises():
    with pytest.raises(ValueError, match="no GID range"):
        check_component_coverage(TWO_BAYS, ranges_frame(["Tail Fin"]))


def test_gid_range_with_no_section_raises():
    with pytest.raises(ValueError, match="no section"):
        check_component_coverage(TWO_BAYS,
                                 ranges_frame(["Fuselage", "Tail Fin"]))


def test_excluded_item_needs_no_sections():
    check_component_coverage(TWO_BAYS,
                             ranges_frame(["Fuselage", "Point Masses"]),
                             exclude_items={"Point Masses"})


def test_whitespace_only_mismatch_is_called_out():
    sec = sections_frame([("RH Wing", "ALL", "a", "RW1", "y", 0.0, 10.0, 0.0,
                           5.0, 0.0)])
    with pytest.raises(ValueError, match="whitespace-only"):
        check_component_coverage(sec, ranges_frame(["RHWing"]))


def test_missing_configuration_sections_are_reported():
    sec = sections_frame([
        ("Fuselage", "FF", "a", "F1", "x", 0.0, 20.0, 10.0, 0.0, 0.0)])
    gaps = check_sections_per_configuration(
        sec, ranges_frame(["Fuselage"]),
        cases_frame([("A", "FF"), ("B", "CC")]))
    assert list(gaps["configuration"]) == ["CC"]


# ------------------------------------------------------------- moments --------

def test_transfer_is_reversible():
    F0, M0 = np.array([1.0, -2.0, 3.0]), np.array([4.0, 5.0, -6.0])
    F1, M1 = transfer(F0, M0, [10.0, 1.0, 2.0], [3.0, -4.0, 5.0])
    F2, M2 = transfer(F1, M1, [3.0, -4.0, 5.0], [10.0, 1.0, 2.0])
    assert np.allclose(F2, F0) and np.allclose(M2, M0)


def two_partitions():
    """Fuselage and wing per component, vehicle bands spanning both."""
    sec = sections_frame([
        ("Fuselage", "ALL", "f", "F1", "x", 0.0, 40.0, 20.0, 0.0, 0.0),
        ("RH Wing", "ALL", "rw", "RW1", "y", 10.0, 40.0, 25.0, 25.0, 0.0),
        ("Vehicle", "ALL", "v1", "V1", "x", 0.0, 25.0, 12.5, 0.0, 0.0),
        ("Vehicle", "ALL", "v2", "V2", "x", 25.0, 40.0, 32.5, 0.0, 0.0)])
    wing = pd.DataFrame({
        "node_id": [2000], "x": [22.0], "y": [25.0], "z": 0.0, "fx": 0.0,
        "fy": 0.0, "fz": 40.0, "loadcase": "LC1", "component": "RH Wing",
        "comp_key": "rh wing", "excluded": False})
    df = pd.concat([nodes_frame([5.0, 30.0]), wing], ignore_index=True)
    body = assign_sections(df, sec, "CC", "Vehicle")
    return sec, body, section_loads(body, sec, "CC", ["LC1"], "Vehicle")


def test_vehicle_sections_bin_every_node():
    _, body, _ = two_partitions()
    assert body["section_id"].notna().all()
    assert body["vehicle_id"].notna().all()
    wing = body[body["comp_key"] == "rh wing"].iloc[0]
    assert wing["section_id"] == "RW1" and wing["vehicle_id"] == "V1"


def test_partitions_agree_at_the_cg():
    _, _, sec = two_partitions()
    assert check_partitions_agree(sec, CG)["ok"]


def test_a_gap_in_the_vehicle_partition_is_caught():
    sec, _, _ = two_partitions()
    sec = sec[sec["section_id"] != "V2"]
    body = assign_sections(
        pd.concat([nodes_frame([5.0, 30.0])], ignore_index=True), sec, "CC",
        "Vehicle")
    out = section_loads(body, sec, "CC", ["LC1"], "Vehicle")
    assert not check_partitions_agree(out, CG)["ok"]


def test_section_nodes_reproduces_its_breakdown_row():
    """At the CG the centroid cancels, so a section equals its nodes summed."""
    _, body, sec = two_partitions()
    pm = pd.DataFrame(columns=["loadcase"] + XCOLS + FCOLS)
    bd = cg_breakdown(sec, pm, CG)
    for sid in bd.query("level == 'section' and partition == 'component'")[
            "section_id"]:
        n = section_nodes({"CC": body}, "LC1", sid)
        F = n[FCOLS].to_numpy(float)
        got = np.concatenate([F.sum(axis=0),
                              np.cross(n[XCOLS].to_numpy(float) - CG, F)
                              .sum(axis=0)])
        assert np.allclose(got, bd[bd["section_id"] == sid][SIX].to_numpy()[0])


# ------------------------------------------------------------- breakdown ------

def breakdown_setup():
    _, _, sec = two_partitions()
    pm = pd.DataFrame({"loadcase": ["LC1"], "x": [10.0], "y": [0.0],
                       "z": [0.0], "fx": [0.0], "fy": [0.0], "fz": [-5.0]})
    return cg_breakdown(sec, pm, CG)


def test_subtotals_sum_their_sections():
    bd = breakdown_setup()
    for comp, g in bd[bd["partition"] == "component"].groupby("component"):
        if not comp:
            continue
        assert np.allclose(g[g["level"] == "section"][SIX].to_numpy().sum(axis=0),
                           g[g["level"] == "subtotal"][SIX].to_numpy()[0])


def test_total_is_subtotals_plus_point_masses():
    g = breakdown_setup().query("partition == 'component'")
    assert np.allclose(
        g[g["level"] == "subtotal"][SIX].to_numpy().sum(axis=0)
        + g[g["level"] == "point masses"][SIX].to_numpy()[0],
        g[g["level"] == "TOTAL"][SIX].to_numpy()[0])


def test_partition_totals_match():
    tot = breakdown_setup().query("level == 'TOTAL'").set_index("partition")
    assert np.allclose(tot.loc["component", SIX].to_numpy(float),
                       tot.loc["vehicle", SIX].to_numpy(float))


def test_summary_rows_come_last():
    """Their component is blank, so sorting on it alone puts them first."""
    levels = list(breakdown_setup().query("partition == 'component'")["level"])
    assert levels[-2:] == ["point masses", "TOTAL"]


# ----------------------------------------------------------------- audit ------

def test_audit_total_is_sections_plus_point_masses():
    df = nodes_frame([2.0, 8.0])
    pm = pd.DataFrame({"loadcase": ["LC1"], "x": [10.0], "y": [0.0],
                       "z": [0.0], "fx": [0.0], "fy": [0.0], "fz": [-20.0]})
    au = force_audit(df, pm, CG).set_index("group")
    assert np.allclose(au.loc["3 card rows in sections", SIX].to_numpy(float)
                       + au.loc["4 point mass loads", SIX].to_numpy(float),
                       au.loc["5 TOTAL (3 + 4)", SIX].to_numpy(float))


def test_audit_separates_the_held_out_rows():
    df = nodes_frame([2.0, 8.0])
    df.loc[1, "excluded"] = True
    pm = pd.DataFrame(columns=["loadcase"] + XCOLS + FCOLS)
    au = force_audit(df, pm, CG).set_index("group")
    assert au.loc["1 all card rows", "Fz"] == pytest.approx(20.0)
    assert au.loc["2 held-out item rows", "Fz"] == pytest.approx(10.0)
    assert au.loc["3 card rows in sections", "Fz"] == pytest.approx(10.0)


def test_balanced_forces_make_the_moment_reference_independent():
    """So a residual moment cannot be blamed on the CG."""
    df = pd.DataFrame({
        "node_id": [1, 2], "x": [0.0, 10.0], "y": 0.0, "z": 0.0, "fx": 0.0,
        "fy": 0.0, "fz": [5.0, -5.0], "loadcase": "LC1", "excluded": False})
    pm = pd.DataFrame(columns=["loadcase"] + XCOLS + FCOLS)
    pick = lambda cg: (force_audit(df, pm, cg).set_index("group")
                       .loc["5 TOTAL (3 + 4)", SIX].to_numpy(float))
    a, b = pick(np.zeros(3)), pick(np.array([999.0, -50.0, 7.0]))
    assert np.allclose(a, b) and not np.allclose(a[3:], 0.0)


# ---------------------------------------------------------------- export ------

def test_exports_are_flat_with_loadcase_first(tmp_path):
    _, _, sec = two_partitions()
    pm = point_mass_loads(
        pd.DataFrame({"node_id": [1], "mass": [1.0], "x": [0.0], "y": [0.0],
                      "z": [0.0]}), cases_frame([("LC1", "CC")]))
    for path, frame in ((export_section_loads(sec, tmp_path / "s.csv"), None),
                        (export_point_mass_loads(pm, tmp_path / "p.csv"), None)):
        out = pd.read_csv(path)
        assert list(out.columns)[0] == "loadcase"
        assert set(SIX if path.name == "s.csv" else FCOLS) <= set(out.columns)


def diagram_setup():
    sec = sections_frame([
        ("Fuselage", "ALL", "a", "F1", "x", 0.0, 20.0, 10.0, 0.0, 0.0),
        ("Fuselage", "ALL", "b", "F2", "x", 20.0, 40.0, 30.0, 0.0, 0.0)])
    body = assign_sections(nodes_frame([10.0, 30.0]), sec, "CC")
    return section_loads(body, sec, "CC", ["LC1"])


def test_diagram_accumulates_and_anchors_at_zero():
    d = station_diagram(diagram_setup()).sort_values("station")
    assert d["Fz"].tolist() == pytest.approx([0.0, 10.0, 20.0])
    assert d["sections"].tolist() == [0, 1, 2]


def test_diagram_direction_reverses_the_accumulation():
    d = station_diagram(diagram_setup(), direction="high").sort_values("station")
    assert d["Fz"].tolist() == pytest.approx([20.0, 10.0, 0.0])


def test_diagram_last_cut_equals_the_component_total():
    sec = diagram_setup()
    d = station_diagram(sec)
    assert d["Fz"].abs().max() == pytest.approx(sec["Fz"].sum())


def test_bad_direction_raises():
    with pytest.raises(ValueError, match="direction"):
        station_diagram(diagram_setup(), direction="outboard")


def test_lat_dir_matching_sta_dir_is_rejected_in_the_tables(tmp_path):
    sec = sections_frame([
        ("Fuselage", "ALL", "a", "F1", "x", 0.0, 20.0, 10.0, 0.0, 0.0)])
    sec["lat_dir"] = "x"
    write_tables(tmp_path, sec)
    with pytest.raises(ValueError, match="cannot equal sta_dir"):
        read_tables(tmp_path)


def test_mixed_lat_dir_within_a_component_is_rejected(tmp_path):
    sec = sections_frame([
        ("Fuselage", "ALL", "a", "F1", "x", 0.0, 20.0, 10.0, 0.0, 0.0),
        ("Fuselage", "ALL", "b", "F2", "x", 20.0, 40.0, 30.0, 0.0, 0.0)])
    sec.loc[1, "lat_dir"] = "z"
    write_tables(tmp_path, sec)
    with pytest.raises(ValueError, match="mixed lat_dir"):
        read_tables(tmp_path)


@pytest.mark.parametrize("axis,lat,lat2", [
    ("x", "y", "z"), ("y", "x", "z"), ("z", "x", "y"), ("x", "z", "y")])
def test_second_lateral_is_derived(axis, lat, lat2):
    """sta_dir and lat_dir fix the frame; the third axis follows."""
    ax = station_axes(axis, lat)
    assert ax["normal"] == axis and ax["lateral 1"] == lat
    assert ax["lateral 2"] == lat2


def test_lat_dir_equal_to_the_station_axis_is_rejected():
    with pytest.raises(ValueError, match="cannot be the station axis"):
        station_axes("z", "z")


def test_columns_are_forces_then_moments_normal_first():
    assert station_columns("y", "x") == [
        ("Fy", "normal"), ("Fx", "lateral 1"), ("Fz", "lateral 2"),
        ("My", "normal"), ("Mx", "lateral 1"), ("Mz", "lateral 2")]


def diagram_two_configs():
    sec = sections_frame([
        ("Fuselage", "CC", "a", "F1", "x", 0.0, 40.0, 20.0, 0.0, 0.0),
        ("Fuselage", "FF", "b", "F2", "x", 0.0, 20.0, 10.0, 0.0, 0.0),
        ("Fuselage", "FF", "c", "F3", "x", 20.0, 40.0, 30.0, 0.0, 0.0)])
    cases = cases_frame([("LC1", "CC"), ("LC2", "FF")])
    df = pd.concat([nodes_frame([10.0, 30.0], loadcase="LC1"),
                    nodes_frame([10.0, 30.0], loadcase="LC2")],
                   ignore_index=True)
    sec_out, _ = run_sections(df, sec, cases)
    return station_diagram(sec_out)


def test_diagram_keeps_configurations_apart():
    d = diagram_two_configs()
    assert set(d["configuration"]) == {"CC", "FF"}
    assert d[d["configuration"] == "CC"]["station"].nunique() == 2
    assert d[d["configuration"] == "FF"]["station"].nunique() == 3


def test_plot_refuses_an_ambiguous_component():
    """Two configurations on shared axes would mix different section grids."""
    import matplotlib
    matplotlib.use("Agg")
    with pytest.raises(ValueError, match="pass configuration="):
        plot_station_diagram(diagram_two_configs(), "Fuselage")


def test_plot_writes_one_png_per_configuration(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    plot_all_station_diagrams(diagram_two_configs(), out_dir=tmp_path)
    names = sorted(p.name for p in tmp_path.glob("*.png"))
    assert names == ["stations_fuselage_cc.png", "stations_fuselage_ff.png"]


def test_drivers_report_configuration():
    d = diagram_two_configs()
    out = station_drivers(d)
    assert set(out["configuration"]) == {"CC", "FF"}
    assert set(out["axis"]) == set(ROLES)


def diagram_with_vehicle(**kw):
    sec = sections_frame([
        ("Fuselage", "ALL", "f", "F1", "x", 0.0, 20.0, 10.0, 0.0, 0.0),
        ("Fuselage", "ALL", "g", "F2", "x", 20.0, 40.0, 30.0, 0.0, 0.0),
        ("Vehicle", "ALL", "v", "V1", "x", 0.0, 40.0, 20.0, 0.0, 0.0)])
    body = assign_sections(nodes_frame([10.0, 30.0]), sec, "CC", "Vehicle")
    return station_diagram(section_loads(body, sec, "CC", ["LC1"], "Vehicle"),
                           **kw)


def test_diagram_leaves_the_vehicle_out_by_default():
    """It covers the same load as the component partition."""
    d = diagram_with_vehicle()
    assert set(d["partition"]) == {"component"}
    assert "Vehicle" not in set(d["component"])


def test_vehicle_partition_available_on_request():
    d = diagram_with_vehicle(partition="vehicle")
    assert set(d["component"]) == {"Vehicle"}


def test_partition_can_be_restricted():
    sec = sections_frame([
        ("Fuselage", "ALL", "f", "F1", "x", 0.0, 40.0, 20.0, 0.0, 0.0),
        ("Vehicle", "ALL", "v", "V1", "x", 0.0, 40.0, 20.0, 0.0, 0.0)])
    body = assign_sections(nodes_frame([10.0, 30.0]), sec, "CC", "Vehicle")
    loads = section_loads(body, sec, "CC", ["LC1"], "Vehicle")
    assert set(station_diagram(loads, partition="vehicle")["component"]) == {
        "Vehicle"}


def test_vehicle_total_matches_the_component_total():
    """Both partitions bin the same nodes, so the last cut must agree."""
    d = diagram_with_vehicle(partition=None)
    last = d.sort_values("station").groupby("partition")["Fz"].last()
    assert last["vehicle"] == pytest.approx(last["component"])


def test_plots_go_into_the_requested_subfolder(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    images = tmp_path / "images"
    plot_all_station_diagrams(diagram_with_vehicle(), out_dir=images)
    assert images.is_dir()
    assert sorted(p.name for p in images.glob("*.png")) == [
        "stations_fuselage_all.png"]


def test_drivers_report_the_partition():
    out = station_drivers(diagram_with_vehicle(partition=None))
    assert set(out["partition"]) == {"component", "vehicle"}


# ================== malformed force card files ================================

def write_cards(root, sub, stem, lines):
    d = root / sub
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stem}.txt").write_text("$h\n$h\nhdr\n" + "\n".join(lines) + "\n")
    return root


ONE_CASE = cases_frame([("LC101", "CC")])


def test_non_numeric_card_value_is_caught_at_read(tmp_path):
    """Otherwise it surfaces much later as a float conversion error."""
    root = write_cards(tmp_path, "CC_Force_Cards", "run_LC101_x",
                       ["1000,0,0,0,0,0,1", "1001,0,0,0,0,0,****"])
    with pytest.raises(ValueError, match="non-numeric 'fz'"):
        read_force_cards(root, ONE_CASE)


def test_header_row_read_as_data_is_caught(tmp_path):
    """The symptom of a wrong skiprows."""
    d = tmp_path / "CC_Force_Cards"
    d.mkdir(parents=True)
    (d / "run_LC101_x.txt").write_text(
        "$h\n$h\nnode_id,x,y,z,fx,fy,fz\n1000,0,0,0,0,0,1\n")
    with pytest.raises(ValueError, match="non-numeric"):
        read_force_cards(tmp_path, ONE_CASE, skiprows=2)


def test_duplicate_node_id_in_one_file_is_caught(tmp_path):
    """A repeated grid point would be counted twice."""
    root = write_cards(tmp_path, "CC_Force_Cards", "run_LC101_x",
                       ["1000,0,0,0,0,0,1", "1000,0,0,0,0,0,1"])
    with pytest.raises(ValueError, match="node_id repeated"):
        read_force_cards(root, ONE_CASE)


def test_load_case_with_no_card_file_is_caught(tmp_path):
    """It would otherwise produce a full set of zero sections."""
    root = write_cards(tmp_path, "CC_Force_Cards", "run_LC101_x",
                       ["1000,0,0,0,0,0,1"])
    cases = cases_frame([("LC101", "CC"), ("LC102", "CC")])
    with pytest.raises(ValueError, match="no card file"):
        read_force_cards(root, cases)


def test_card_file_with_no_sheet_row_is_caught(tmp_path):
    """Caught while resolving the filename, since the LC token is unknown."""
    root = write_cards(tmp_path, "CC_Force_Cards", "run_LC101_x",
                       ["1000,0,0,0,0,0,1"])
    write_cards(root, "CC_Force_Cards", "run_LC102_x", ["1001,0,0,0,0,0,1"])
    with pytest.raises(ValueError, match="matches no loadcase"):
        read_force_cards(root, ONE_CASE)


def test_no_card_files_at_all_is_caught(tmp_path):
    (tmp_path / "CC_Force_Cards").mkdir()
    with pytest.raises(FileNotFoundError):
        read_force_cards(tmp_path, ONE_CASE)


def test_filename_without_an_lc_token_is_caught(tmp_path):
    root = write_cards(tmp_path, "CC_Force_Cards", "run_final",
                       ["1000,0,0,0,0,0,1"])
    with pytest.raises(ValueError, match="no LC"):
        read_force_cards(root, ONE_CASE)


# ================== malformed tables =========================================

def test_missing_section_column_names_the_file(tmp_path):
    sec = sections_frame([
        ("Fuselage", "ALL", "a", "F1", "x", 0.0, 20.0, 10.0, 0.0, 0.0)])
    write_tables(tmp_path, sec.drop(columns="cen_z"))
    with pytest.raises(KeyError, match="sections.csv missing"):
        read_tables(tmp_path)


def test_zero_width_section_is_caught(tmp_path):
    sec = sections_frame([
        ("Fuselage", "ALL", "a", "F1", "x", 20.0, 20.0, 20.0, 0.0, 0.0)])
    write_tables(tmp_path, sec)
    with pytest.raises(ValueError, match="zero-width"):
        read_tables(tmp_path)


def test_gid_ranges_overlapping_between_items_is_caught(tmp_path):
    sec = sections_frame([
        ("Fuselage", "ALL", "a", "F1", "x", 0.0, 20.0, 10.0, 0.0, 0.0)])
    write_tables(tmp_path, sec)
    pd.DataFrame({"gid_start": [1000, 1500], "gid_end": [1999, 2499],
                  "item": ["Fuselage", "RH Wing"]}).to_csv(
        tmp_path / "node_ranges.csv", index=False)
    with pytest.raises(ValueError, match="GID ranges overlap"):
        read_tables(tmp_path)


def test_reversed_gid_range_is_caught():
    rng = pd.DataFrame({"gid_start": [1999], "gid_end": [1000],
                        "item": ["Fuselage"], "comp_key": ["fuselage"]})
    df = pd.DataFrame({"node_id": [1500], "x": 0.0, "y": 0.0, "z": 0.0,
                       "fx": 0.0, "fy": 0.0, "fz": 1.0, "loadcase": "LC1"})
    with pytest.raises(ValueError, match="gid_end < gid_start"):
        tag_components(df, rng)


def ranges_csv(d, rows):
    pd.DataFrame(rows, columns=["gid_start", "gid_end", "item"]).to_csv(
        d / "node_ranges.csv", index=False)


def test_several_ranges_per_component_are_allowed(tmp_path):
    """An inboard/outboard split is a normal way to describe one component."""
    sec = sections_frame([
        ("Fuselage", "ALL", "a", "F1", "x", 0.0, 20.0, 10.0, 0.0, 0.0)])
    write_tables(tmp_path, sec)
    ranges_csv(tmp_path, [(1000, 1499, "Fuselage"), (1600, 1999, "Fuselage")])
    _, nr, _, _ = read_tables(tmp_path)
    assert len(nr) == 2 and set(nr["comp_key"]) == {"fuselage"}


def test_overlap_hidden_by_a_same_component_row_is_caught(tmp_path):
    """A broad range overlapping a later one, with a same-item row between.
    An adjacent-pair sweep misses this; the nodes then go to whichever item
    appears later in the file."""
    sec = sections_frame([
        ("Fuselage", "ALL", "a", "F1", "x", 0.0, 20.0, 10.0, 0.0, 0.0),
        ("Tail Fin", "ALL", "b", "T1", "z", 0.0, 20.0, 0.0, 0.0, 10.0)])
    write_tables(tmp_path, sec)
    ranges_csv(tmp_path, [(1000, 9000, "Fuselage"), (2000, 2100, "Fuselage"),
                          (8000, 8100, "Tail Fin")])
    with pytest.raises(ValueError, match="Tail Fin"):
        read_tables(tmp_path)


def test_touching_ranges_of_different_components_are_caught(tmp_path):
    """Shared endpoint: node 1999 would belong to both."""
    sec = sections_frame([
        ("Fuselage", "ALL", "a", "F1", "x", 0.0, 20.0, 10.0, 0.0, 0.0),
        ("Tail Fin", "ALL", "b", "T1", "z", 0.0, 20.0, 0.0, 0.0, 10.0)])
    write_tables(tmp_path, sec)
    ranges_csv(tmp_path, [(1000, 1999, "Fuselage"), (1999, 2999, "Tail Fin")])
    with pytest.raises(ValueError, match="overlap"):
        read_tables(tmp_path)


def test_nodes_split_across_two_ranges_of_one_component():
    """Both ranges must land in the same component and bin normally."""
    rng = pd.DataFrame({"gid_start": [1000, 1600], "gid_end": [1499, 1999],
                        "item": "Fuselage", "comp_key": "fuselage"})
    df = pd.DataFrame({"node_id": [1100, 1700], "x": [10.0, 30.0], "y": 0.0,
                       "z": 0.0, "fx": 0.0, "fy": 0.0, "fz": 5.0,
                       "loadcase": "LC1"})
    out = tag_components(df, rng)
    assert list(out["component"]) == ["Fuselage", "Fuselage"]
    body = assign_sections(out, TWO_BAYS, "CC")
    assert body["section_id"].tolist() == ["F1", "F2"]


# ================== exclude_nodes ============================================

@pytest.mark.parametrize("cell,expect", [
    ("1000,1001", {1000, 1001}),
    ("1000; 1001", {1000, 1001}),
    ("1000 1001", {1000, 1001}),
    ("2000-2003", {2000, 2001, 2002, 2003}),
    ("1000, 2000-2002 3000", {1000, 2000, 2001, 2002, 3000}),
    ("", set()), (None, set()), (float("nan"), set()),
])
def test_node_list_parsing(cell, expect):
    assert parse_node_list(cell) == expect


@pytest.mark.parametrize("bad", ["abc", "1000-990", "10.5"])
def test_bad_node_list_raises(bad):
    with pytest.raises(ValueError):
        parse_node_list(bad)


def test_exclude_nodes_column_is_optional(tmp_path):
    sec = sections_frame([
        ("Fuselage", "ALL", "a", "F1", "x", 0.0, 20.0, 10.0, 0.0, 0.0)])
    write_tables(tmp_path, sec)
    _, _, lcs, _ = read_tables(tmp_path)
    assert list(lcs["exclude_nodes"]) == [set()]


def test_exclude_nodes_column_is_parsed(tmp_path):
    sec = sections_frame([
        ("Fuselage", "ALL", "a", "F1", "x", 0.0, 20.0, 10.0, 0.0, 0.0)])
    write_tables(tmp_path, sec)
    lc = pd.read_csv(tmp_path / "load_cases.csv")
    lc["exclude_nodes"] = "1000, 2000-2002"
    lc.to_csv(tmp_path / "load_cases.csv", index=False)
    _, _, lcs, _ = read_tables(tmp_path)
    assert lcs["exclude_nodes"].iloc[0] == {1000, 2000, 2001, 2002}


def test_exclude_nodes_drops_a_point_mass_for_that_case_only():
    """The exclusion covers point mass ids as well as card ids."""
    pm = pd.DataFrame({"node_id": [9001, 9002], "mass": [10.0, 40.0],
                       "x": 0.0, "y": 0.0, "z": 0.0})
    lcs = cases_frame([("LC1", "CC"), ("LC2", "CC")])
    lcs["exclude_nodes"] = [{9002}, set()]
    out = point_mass_loads(pm, lcs)
    assert set(out.loc[out["loadcase"] == "LC1", "node_id"]) == {9001}
    assert set(out.loc[out["loadcase"] == "LC2", "node_id"]) == {9001, 9002}
    assert out.loc[out["loadcase"] == "LC1", "fz"].sum() == pytest.approx(-10.0)
    assert out.loc[out["loadcase"] == "LC2", "fz"].sum() == pytest.approx(-50.0)


def test_excluding_every_point_mass_gives_no_load():
    pm = pd.DataFrame({"node_id": [9001], "mass": [10.0], "x": 0.0, "y": 0.0,
                       "z": 0.0})
    lcs = cases_frame([("LC1", "CC")])
    lcs["exclude_nodes"] = [{9001}]
    assert point_mass_loads(pm, lcs).empty


@pytest.mark.parametrize("cell,expect", [
    ("9002.0", {9002}), (9002.0, {9002}), ("1000.0, 2000-2002",
                                           {1000, 2000, 2001, 2002})])
def test_ids_read_as_float_are_accepted(cell, expect):
    """A column holding one number is read as float, so 9002 arrives 9002.0."""
    assert parse_node_list(cell) == expect


def test_exclusion_report_flags_an_unknown_id():
    df = pd.DataFrame({"node_id": [1000], "loadcase": "LC1", "x": 0.0,
                       "y": 0.0, "z": 0.0, "fx": 0.0, "fy": 0.0, "fz": 1.0})
    pm = pd.DataFrame({"node_id": [9001], "mass": [1.0]})
    lcs = cases_frame([("LC1", "CC")])
    lcs["exclude_nodes"] = [{1000, 9001, 12345}]
    r = exclusion_report(df, pm, lcs).iloc[0]
    assert r["matched_card"] == 1 and r["matched_point_mass"] == 1
    assert r["unknown"] == [12345]


# ================== CSV quoting ==============================================

def write_cases(d, line):
    for f in ("sections.csv", "node_ranges.csv", "point_masses.csv"):
        pass
    (d / "load_cases.csv").write_text(
        "loadcase,configuration,nx,ny,nz,case_description,exclude_nodes\n"
        + line + "\n")


def tables_with_cases(tmp_path, line):
    sec = sections_frame([
        ("Fuselage", "ALL", "a", "F1", "x", 0.0, 20.0, 10.0, 0.0, 0.0)])
    write_tables(tmp_path, sec)
    write_cases(tmp_path, line)
    return read_tables(tmp_path)


@pytest.mark.parametrize("cell,expect", [
    ('"46270, 46271"', {46270, 46271}),          # properly quoted
    (' "46270, 46271"', {46270, 46271}),         # space before the quote
    ('"""46270, 46271"""', {46270, 46271}),      # doubled by Excel
    ("46270;46271", {46270, 46271}),             # semicolons, no quoting needed
    ("46270 46271", {46270, 46271}),             # spaces
    ("46270-46272", {46270, 46271, 46272}),      # range
])
def test_exclude_nodes_survives_csv_quoting(tmp_path, cell, expect):
    """Quote characters can survive the read, and a space before the quote
    changes how pandas parses the cell."""
    _, _, lcs, _ = tables_with_cases(tmp_path, f"101,CC,0,0,1,a,{cell}")
    assert lcs["exclude_nodes"].iloc[0] == expect


def test_unquoted_comma_list_raises_rather_than_losing_data(tmp_path):
    """More fields than headers shifts or drops columns; pandas only warns."""
    with pytest.raises(ValueError, match="more fields than the header"):
        tables_with_cases(tmp_path, "101,CC,0,0,1,a,46270, 46271")


def test_apply_exclusions_removes_card_rows_for_that_case_only():
    df = pd.concat([nodes_frame([1.0, 2.0], loadcase="LC1"),
                    nodes_frame([1.0, 2.0], loadcase="LC2")], ignore_index=True)
    df["node_id"] = [1000, 1001, 1000, 1001]
    lcs = cases_frame([("LC1", "CC"), ("LC2", "CC")])
    lcs["exclude_nodes"] = [{1001}, set()]
    out, rep = apply_exclusions(df, lcs)
    assert sorted(out.loc[out["loadcase"] == "LC1", "node_id"]) == [1000]
    assert sorted(out.loc[out["loadcase"] == "LC2", "node_id"]) == [1000, 1001]
    r = rep.set_index("loadcase").loc["LC1"]
    assert r["rows_removed"] == 1 and r["not_present"] == 0


def test_apply_exclusions_counts_ids_that_were_not_there():
    """A declared id absent from the cards is reported, not an error -- it may
    be a point mass id, which point_mass_loads handles."""
    df = nodes_frame([1.0]).assign(node_id=1000)
    lcs = cases_frame([("LC1", "CC")])
    lcs["exclude_nodes"] = [{1000, 9001}]
    out, rep = apply_exclusions(df, lcs)
    assert out.empty
    assert rep["not_present"].iloc[0] == 1


def test_apply_exclusions_is_a_no_op_without_the_column():
    df = nodes_frame([1.0, 2.0])
    lcs = cases_frame([("LC1", "CC")]).drop(columns=[], errors="ignore")
    out, rep = apply_exclusions(df, lcs)
    assert len(out) == len(df) and rep.empty


def many_case_diagram(n_cases=20):
    sec = sections_frame([
        ("Fuselage", "ALL", "a", "F1", "x", 0.0, 20.0, 10.0, 0.0, 0.0),
        ("Fuselage", "ALL", "b", "F2", "x", 20.0, 40.0, 30.0, 0.0, 0.0)])
    frames = []
    for i in range(n_cases):
        d = nodes_frame([10.0, 30.0], loadcase=f"LC{300 + i}", fz=1.0 + i)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    cases = cases_frame([(f"LC{300 + i}", "CC") for i in range(n_cases)])
    out, _ = run_sections(df, sec, cases)
    return station_diagram(out)


def test_panel_drivers_pick_the_extremes():
    d = many_case_diagram(6)
    lead = panel_drivers(d, "Fz", n=2)
    assert "LC305" in lead          # largest fz
    assert "LC300" in lead          # smallest


def test_panel_drivers_returns_every_envelope_case_by_default():
    """Whoever is max or min at any station, so the set can differ per panel
    and per station -- the nose driver is not always the tail driver."""
    d = many_case_diagram(6)
    lead = set(panel_drivers(d, "Fz"))
    for _, g in d.groupby("station"):
        assert g.loc[g["Fz"].idxmax(), "loadcase"] in lead
        assert g.loc[g["Fz"].idxmin(), "loadcase"] in lead


def test_drivers_cap_limits_the_list():
    d = many_case_diagram(6)
    assert len(panel_drivers(d, "Fz", n=1)) == 1


def test_plot_names_only_the_driving_cases(tmp_path):
    """With many cases a per-case legend is unreadable, so only the extremes
    are labelled."""
    import matplotlib
    matplotlib.use("Agg")
    fig = plot_station_diagram(many_case_diagram(20), "Fuselage",
                               out_dir=tmp_path, drivers=3)
    labelled = {t.get_text() for ax in fig.axes for t in
                (ax.get_legend().get_texts() if ax.get_legend() else [])}
    assert 0 < len(labelled) <= 6
    assert (tmp_path / "stations_fuselage_all.png").exists()


def test_highlight_overrides_the_automatic_pick(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    fig = plot_station_diagram(many_case_diagram(20), "Fuselage",
                               highlight=["LC307"], out_dir=tmp_path)
    labelled = {t.get_text() for ax in fig.axes for t in
                (ax.get_legend().get_texts() if ax.get_legend() else [])}
    assert labelled == {"LC307"}


def test_negligible_panels_are_marked(tmp_path):
    """A panel at 1e-13 against another at 1e5 is round-off, not load."""
    import matplotlib
    matplotlib.use("Agg")
    fig = plot_station_diagram(many_case_diagram(5), "Fuselage",
                               out_dir=tmp_path)
    titles = [ax.get_title() for ax in fig.axes]
    assert any("(~0)" in t for t in titles), titles