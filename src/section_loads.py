"""Section loads at RBE3 centroids, from nodal force cards.

Conventions:
    sta_dir is x, y or z; the station is the raw global coordinate, so a LH
    component has negative brackets, taken in either order.

    Bins are (sta_lo, sta_hi], with only the lowest section of a component
    closed at the bottom.

    Sections are ALL, FF or CC; load cases are FF or CC. A component declares
    its sections one way or the other, never both.

    The vehicle component's sections bin every body node, forming a second
    partition of the same load. Never summed with the component sections.

    Point mass force is inertia_sign * mass * N, mass in lbm and N in g.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

FCOLS = ["fx", "fy", "fz"]
XCOLS = ["x", "y", "z"]
SIX = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
CARD_COLUMNS = ["node_id", "x", "y", "z", "fx", "fy", "fz"]
SECTION_COLS = ["component", "configuration", "label", "section_id", "sta_dir",
                "lat_dir", "sta_start", "sta_end", "cen_x", "cen_y", "cen_z"]
ALL_CONFIG = "ALL"
STA_DIR_RE = re.compile(r"^\s*([xyz])\s*$", re.I)
LOADCASE_RE = re.compile(r"LC\s*(\d+)", re.I)


def norm(name) -> str:
    """Case- and whitespace-folded key for matching names."""
    return " ".join(str(name).split()).casefold()


def norm_config(value) -> str:
    return " ".join(str(value).split()).upper()


def norm_loadcase(value) -> str:
    """Canonical loadcase key, e.g. 'LC101' from 101 or 'lc101'."""
    t = " ".join(str(value).split()).upper()
    if t.endswith(".0"):
        t = t[:-2]
    return t if t.startswith("LC") else f"LC{t}"


def parse_node_list(value) -> set:
    """Parses a cell like "1000, 1001, 2000-2005" into a set of node ids.

    Accepts commas, semicolons or spaces as separators and `a-b` for an
    inclusive range. Blank or missing gives an empty set.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    # Quote characters can survive the CSV read when a cell was written as
    # "1000, 1001" and the quoting was doubled or preceded by a space.
    text = str(value).strip().strip("\"'").replace('"', " ").replace("'", " ")
    text = text.strip()
    if not text or text.lower() in ("nan", "none"):
        return set()
    out = set()
    for tok in re.split(r"[,;\s]+", text):
        if not tok:
            continue
        # A column holding a single number is read as float, so "9002" arrives
        # as "9002.0".
        tok = re.sub(r"^(\d+)\.0+$", r"\1", tok)
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", tok)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if hi < lo:
                raise ValueError(f"node range '{tok}' is reversed")
            out.update(range(lo, hi + 1))
        elif re.fullmatch(r"\d+", tok):
            out.add(int(tok))
        else:
            raise ValueError(f"'{tok}' in exclude_nodes is not a node id or "
                             "an a-b range")
    return out


def parse_sta_dir(value) -> str:
    m = STA_DIR_RE.match(str(value))
    if not m:
        raise ValueError(f"sta_dir '{value}' is not x, y or z")
    return m.group(1).lower()


def station_of(frame, sta_dir) -> np.ndarray:
    return frame[parse_sta_dir(sta_dir)].to_numpy(float)


def bracket(sta_start, sta_end):
    """(low, high), accepting either fill order."""
    lo, hi = float(sta_start), float(sta_end)
    return (lo, hi) if lo <= hi else (hi, lo)


def canon_headers(frame):
    """Lower-snake-case the column names and drop blank rows."""
    out = frame.rename(columns=lambda c: "_".join(str(c).split()).lower())
    dupes = out.columns[out.columns.duplicated()].tolist()
    if dupes:
        raise ValueError(f"columns {dupes} collide after normalising headers")
    return out.dropna(how="all").reset_index(drop=True)


def infer_vehicle(sections, node_ranges):
    """The component whose sections span every node, not just its own.

    It is the one in sections.csv with no GID range, since it owns no nodes of
    its own. Derived rather than declared, so a name cannot be mistyped in two
    places.

    Returns:
        The component name, or None if there is no such component.
    """
    sec = {norm(c): c for c in sections["component"]}
    owned = {norm(i) for i in node_ranges["item"]}
    spare = sorted(k for k in sec if k not in owned)
    if not spare:
        return None
    if len(spare) > 1:
        raise ValueError(
            f"{[sec[k] for k in spare]} have sections but no GID range, so the "
            "vehicle is ambiguous. Every component except the vehicle needs a "
            "row in node_ranges.csv.")
    return sec[spare[0]]


def read_tables(table_dir):
    """Reads and validates the four CSVs."""
    d = Path(table_dir)
    # index_col=False and skipinitialspace matter here: without them a row with
    # more fields than headers silently shifts every column into the next, and a
    # space before a quoted cell drops part of its contents.
    def read(name):
        with warnings.catch_warnings():
            # A row with more fields than headers loses data. pandas only warns.
            warnings.simplefilter("error", pd.errors.ParserWarning)
            try:
                raw = pd.read_csv(d / name, index_col=False,
                                  skipinitialspace=True)
            except pd.errors.ParserWarning as e:
                raise ValueError(
                    f"{name}: a row has more fields than the header, so data "
                    "would be lost. A cell containing commas must be quoted, "
                    'e.g. "1000, 1001". Semicolons or spaces avoid the issue '
                    f"entirely.\n  pandas said: {e}") from None
        return canon_headers(raw)

    sections = read("sections.csv")
    node_ranges = read("node_ranges.csv")
    load_cases = read("load_cases.csv")
    point_masses = read("point_masses.csv")

    for name, tbl, need in (
            ("sections", sections, SECTION_COLS),
            ("node_ranges", node_ranges, ["gid_start", "gid_end", "item"]),
            ("load_cases", load_cases,
             ["loadcase", "configuration", "nx", "ny", "nz"]),
            ("point_masses", point_masses, ["node_id", "mass"])):
        missing = [c for c in need if c not in tbl.columns]
        if missing:
            raise KeyError(f"{name}.csv missing {missing}; has {list(tbl.columns)}")

    sections["comp_key"] = sections["component"].map(norm)
    sections["config"] = sections["configuration"].map(norm_config)
    node_ranges["comp_key"] = node_ranges["item"].map(norm)
    load_cases["config"] = load_cases["configuration"].map(norm_config)
    load_cases["loadcase"] = load_cases["loadcase"].map(norm_loadcase)
    # Optional: node ids a case is expected to be missing, so a legitimate
    # difference between cases is declared rather than reported as a fault.
    load_cases["exclude_nodes"] = (
        load_cases["exclude_nodes"].map(parse_node_list)
        if "exclude_nodes" in load_cases.columns
        else [set() for _ in range(len(load_cases))])

    for tbl, keys, what in ((sections, ["section_id", "config"],
                             "section_id within a configuration"),
                            (load_cases, ["loadcase"], "loadcase")):
        blank = tbl[keys[0]].isna()
        if blank.any():
            raise ValueError(f"{what}: empty '{keys[0]}' in sheet row(s) "
                             f"{sorted(tbl.index[blank] + 2)}")
        dup = tbl.duplicated(keys, keep=False)
        if dup.any():
            raise ValueError(
                f"duplicate {what}:\n"
                f"{tbl.loc[dup, keys].drop_duplicates().to_string(index=False)}")

    for r in sections.itertuples(index=False):
        if parse_sta_dir(r.lat_dir) == parse_sta_dir(r.sta_dir):
            raise ValueError(f"section '{r.section_id}': lat_dir '{r.lat_dir}' "
                             f"cannot equal sta_dir '{r.sta_dir}'")
    zero = sections[sections["sta_start"] == sections["sta_end"]]
    if len(zero):
        raise ValueError("zero-width sections:\n"
                         f"{zero[['section_id']].to_string(index=False)}")
    los, his = zip(*[bracket(r.sta_start, r.sta_end)
                     for r in sections.itertuples(index=False)])
    sections["sta_lo"], sections["sta_hi"] = los, his

    for (_, cfg), g in sections.groupby(["comp_key", "config"]):
        for col in ("sta_dir", "lat_dir"):
            if len({parse_sta_dir(v) for v in g[col].unique()}) > 1:
                raise ValueError(f"'{g['component'].iloc[0]}' config {cfg} has "
                                 f"mixed {col} {sorted(g[col].unique())}")

    mixed = [(g["component"].iloc[0], sorted(set(g["config"])))
             for _, g in sections.groupby("comp_key")
             if ALL_CONFIG in set(g["config"]) and set(g["config"]) - {ALL_CONFIG}]
    if mixed:
        raise ValueError("components mixing ALL with FF/CC sections:\n"
                         + "\n".join(f"  '{c}' declares {v}" for c, v in mixed))

    # Several ranges per item is fine. Overlap between DIFFERENT items is not,
    # because tagging is last-wins and would silently reassign nodes. Compared
    # pairwise: an adjacent-only sweep misses a broad range overlapping a later
    # one when a same-item row sits between them.
    nr = node_ranges.reset_index(drop=True)
    clashes = []
    for i in range(len(nr)):
        for j in range(i + 1, len(nr)):
            a, b = nr.iloc[i], nr.iloc[j]
            if (a["comp_key"] != b["comp_key"]
                    and a["gid_start"] <= b["gid_end"]
                    and b["gid_start"] <= a["gid_end"]):
                clashes.append(
                    f"'{a['item']}' [{a['gid_start']}, {a['gid_end']}] and "
                    f"'{b['item']}' [{b['gid_start']}, {b['gid_end']}]")
    if clashes:
        raise ValueError("GID ranges overlap between different items:\n  "
                         + "\n  ".join(clashes[:10]))
    return sections, node_ranges, load_cases, point_masses


def loadcase_from_stem(stem, known=None) -> str:
    """Loadcase from a stem like x_x_LC101_x.

    `known` splits a contiguous digit run, where LC10110002 is LC101 + 10002.
    """
    hits = LOADCASE_RE.findall(str(stem))
    if not hits:
        raise ValueError(f"no LC<digits> token in stem '{stem}'")
    if known is None:
        if len(set(hits)) > 1:
            raise ValueError(f"stem '{stem}' has several LC tokens; pass known=")
        return norm_loadcase(hits[0])
    digits = {norm_loadcase(k)[2:]: norm_loadcase(k) for k in known}
    for run in hits:
        for n in sorted({len(d) for d in digits}, reverse=True):
            if run[:n] in digits:
                return digits[run[:n]]
    raise ValueError(f"stem '{stem}' matches no loadcase in load_cases.csv")


def config_from_path(path, root, configs=None):
    """Configuration from the folder, e.g. CC_Force_Cards -> CC."""
    parts = list(Path(path).relative_to(root).parts[:-1])
    if not parts:
        return None
    if configs:
        want = {norm_config(c) for c in configs}
        for part in reversed(parts):
            for tok in re.split(r"[^A-Za-z0-9]+", part):
                if norm_config(tok) in want:
                    return norm_config(tok)
        return None
    return norm_config(re.split(r"[^A-Za-z0-9]+", parts[-1])[0])


def read_force_cards(data_dir, load_cases, only=None, pattern="*.txt",
                     skiprows=3):
    """Reads the force cards. Returns (df, load_cases) narrowed to `only`."""
    root = Path(data_dir)
    files = sorted(root.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"no {pattern} under {root.resolve()}")

    known = load_cases["loadcase"]
    configs = load_cases["config"].unique()
    keep = None
    if only is not None and len(list(only)):
        want = [norm_loadcase(v) for v in only]
        unknown = [k for k in want if k not in set(known)]
        if unknown:
            raise ValueError(f"loadcase(s) {unknown} are not in load_cases.csv")
        keep = set(want)

    frames, seen = [], {}
    for f in files:
        lc = loadcase_from_stem(f.stem, known)
        if keep is not None and lc not in keep:
            continue
        cfg = config_from_path(f, root, configs)
        if (lc, cfg) in seen:
            raise ValueError(f"loadcase '{lc}' config '{cfg}' matched by two "
                             f"files: {seen[(lc, cfg)]} and "
                             f"{f.relative_to(root)}")
        seen[(lc, cfg)] = str(f.relative_to(root))
        d = pd.read_csv(f, skiprows=skiprows, header=None, names=CARD_COLUMNS)
        for col in CARD_COLUMNS:
            bad = pd.to_numeric(d[col], errors="coerce").isna() & d[col].notna()
            if bad.any():
                rows = (d.index[bad] + skiprows + 1).tolist()[:5]
                raise ValueError(
                    f"{f.relative_to(root)}: non-numeric '{col}' on line(s) "
                    f"{rows} (values {d.loc[bad, col].head().tolist()}). Check "
                    "skiprows and the delimiter.")
            d[col] = pd.to_numeric(d[col])
        dup = d["node_id"].duplicated(keep=False)
        if dup.any():
            ids = sorted(set(d.loc[dup, "node_id"]))
            raise ValueError(
                f"{f.relative_to(root)}: node_id repeated {len(ids)} time(s), "
                f"e.g. {ids[:8]}. Those loads would be counted twice.")
        d["loadcase"] = lc
        d["config_folder"] = cfg
        d["source_file"] = str(f.relative_to(root))
        frames.append(d)

    if not frames:
        raise ValueError(f"no card files match {sorted(keep)}")
    df = pd.concat(frames, ignore_index=True)
    out_cases = (load_cases if keep is None
                 else load_cases[load_cases["loadcase"].isin(keep)]
                 .reset_index(drop=True))

    missing = sorted(set(out_cases["loadcase"]) - set(df["loadcase"]))
    if missing:
        raise ValueError(f"load case(s) {missing} are in load_cases.csv but "
                         "have no card file; they would produce zeros")
    extra = sorted(set(df["loadcase"]) - set(out_cases["loadcase"]))
    if extra:
        raise ValueError(f"card file(s) for {extra} have no row in "
                         "load_cases.csv, so no load factors")
    return df, out_cases


def check_card_configurations(df, load_cases):
    """Card folder configuration against load_cases.csv, per loadcase."""
    folder = df.drop_duplicates("loadcase").set_index("loadcase")["config_folder"]
    table = load_cases.set_index("loadcase")["config"]
    out = pd.concat([folder.rename("from_folder"),
                     table.rename("from_csv")], axis=1)
    out["agree"] = out["from_folder"] == out["from_csv"]
    return out.reset_index()


def check_component_coverage(sections, node_ranges, vehicle=None,
                             exclude_items=frozenset()):
    """Bidirectional component match between sections and node_ranges."""
    sec_map = (sections.drop_duplicates("comp_key")
               .set_index("comp_key")["component"].to_dict())
    rng_map = (node_ranges.drop_duplicates("comp_key")
               .set_index("comp_key")["item"].to_dict())
    exempt = {norm(x) for x in exclude_items}
    if vehicle:
        exempt.add(norm(vehicle))
    no_range = set(sec_map) - set(rng_map) - exempt
    no_section = set(rng_map) - set(sec_map) - exempt

    msgs = []
    if no_range:
        msgs.append("in sections.csv with no GID range: "
                    + ", ".join(f"'{sec_map[c]}'" for c in sorted(no_range)))
    if no_section:
        msgs.append("in node_ranges.csv with no section: "
                    + ", ".join(f"'{rng_map[c]}'" for c in sorted(no_section)))
    if msgs:
        near = [f"'{sec_map[a]}' vs '{rng_map[b]}'"
                for a in sorted(no_range) for b in sorted(no_section)
                if a.replace(" ", "") == b.replace(" ", "")]
        text = "component coverage mismatch:\n  " + "\n  ".join(msgs)
        if near:
            text += "\n  whitespace-only difference: " + "; ".join(near)
        raise ValueError(text)
    return {"no_range": no_range, "no_section": no_section}


def check_configurations(sections, load_cases):
    """Load case configurations against the ones sections declare."""
    sec_cfgs = set(sections["config"])
    case_cfgs = set(load_cases["config"])
    if ALL_CONFIG in case_cfgs:
        raise ValueError(f"load case configuration '{ALL_CONFIG}' is not "
                         "meaningful; a case belongs to one, e.g. FF or CC")
    return {"unserved": case_cfgs - sec_cfgs,
            "unused": sec_cfgs - case_cfgs - {ALL_CONFIG}}


def effective_sections(sections, config):
    """Sections applying to one configuration, resolved per component."""
    cfg = norm_config(config)
    out = []
    for _, g in sections.groupby("comp_key", sort=False):
        cfgs = set(g["config"])
        take = g[g["config"] == (ALL_CONFIG if ALL_CONFIG in cfgs else cfg)]
        if len(take):
            out.append(take)
    return (pd.concat(out, ignore_index=True) if out
            else sections.iloc[0:0].copy())


def check_sections_per_configuration(sections, node_ranges, load_cases,
                                     vehicle=None, exclude_items=frozenset()):
    """Items with no sections in a configuration that has load cases."""
    exempt = {norm(x) for x in exclude_items}
    if vehicle:
        exempt.add(norm(vehicle))
    items = (node_ranges.drop_duplicates("comp_key")
             .set_index("comp_key")["item"].to_dict())
    gaps = [{"configuration": cfg, "component": name}
            for cfg in sorted(set(load_cases["config"]))
            for ck, name in items.items()
            if ck not in exempt
            and ck not in set(effective_sections(sections, cfg)["comp_key"])]
    return pd.DataFrame(gaps)


def tag_components(df, node_ranges, exclude_items=frozenset()):
    """Tags each node with its node_ranges item and flags what to hold out."""
    df = df.copy()
    comp = pd.Series("UNTAGGED", index=df.index, dtype=object)
    for r in node_ranges.itertuples(index=False):
        if r.gid_end < r.gid_start:
            raise ValueError(f"node range '{r.item}' has gid_end < gid_start")
        comp[df["node_id"].between(r.gid_start, r.gid_end)] = r.item
    df["component"] = comp
    df["comp_key"] = df["component"].map(norm)
    df["excluded"] = df["comp_key"].isin({norm(i) for i in exclude_items})
    return df


def point_mass_loads(point_masses, load_cases, inertia_sign=-1.0,
                     accel_to_g=1.0):
    """F = inertia_sign * mass * N, per point mass per load case.

    A node id listed in a case's exclude_nodes is dropped for that case, since
    the exclusion covers point mass ids as well as card ids. The per-case mass
    total is checked against the per-case sum, so a dropped mass cannot be
    silently reapplied.
    """
    pm = point_masses.copy()
    missing = [c for c in XCOLS if c not in pm.columns]
    if missing:
        raise KeyError(f"point_masses.csv missing {missing}")

    acc = load_cases.set_index("loadcase")[["nx", "ny", "nz"]] / accel_to_g
    drops = (load_cases.set_index("loadcase")["exclude_nodes"].to_dict()
             if "exclude_nodes" in load_cases.columns else {})

    out = []
    for lc, a in acc.iterrows():
        drop = drops.get(lc) or set()
        g = pm[~pm["node_id"].isin(drop)].copy()
        g["loadcase"] = lc
        for col, ax in zip(FCOLS, ["nx", "ny", "nz"]):
            g[col] = inertia_sign * g["mass"] * a[ax]
        out.append(g)
    res = pd.concat(out, ignore_index=True)

    for lc, a in acc.iterrows():
        drop = drops.get(lc) or set()
        total = pm.loc[~pm["node_id"].isin(drop), "mass"].sum()
        for col, ax in zip(FCOLS, ["nx", "ny", "nz"]):
            want = inertia_sign * total * a[ax]
            got = res.loc[res["loadcase"] == lc, col].sum()
            if abs(got - want) > 1e-9 * max(abs(want), 1.0):
                raise AssertionError(f"{lc} {col}: {got} != {want}")
    return res


def exclusion_report(df, point_masses, load_cases):
    """What each case's exclude_nodes actually matched.

    An id matching neither a card node nor a point mass does nothing, which is
    usually a typo.
    """
    if "exclude_nodes" not in load_cases.columns:
        return pd.DataFrame()
    card_ids = set(df["node_id"])
    pm_ids = set(point_masses["node_id"])
    rows = []
    for r in load_cases.itertuples(index=False):
        drop = r.exclude_nodes or set()
        present = set(df.loc[df["loadcase"] == r.loadcase, "node_id"])
        rows.append({
            "loadcase": r.loadcase, "declared": len(drop),
            "matched_card": len(drop & card_ids),
            "matched_point_mass": len(drop & pm_ids),
            "still_present_in_cards": len(drop & present),
            "unknown": sorted(drop - card_ids - pm_ids)[:12]})
    return pd.DataFrame(rows)


def assign_sections(df, sections, config, vehicle=None):
    """Bins body nodes for one configuration into section_id and vehicle_id."""
    eff = effective_sections(sections, config)
    vkey = norm(vehicle) if vehicle else None
    partitions = ((eff[eff["comp_key"] != vkey], "section_id", "component"),
                  (eff[eff["comp_key"] == vkey], "vehicle_id", "vehicle"))

    body = df[~df["excluded"]].copy()
    for _, col, _ in partitions:
        body[col] = pd.Series(pd.NA, index=body.index, dtype=object)

    for eff_set, col, scope in partitions:
        if not len(eff_set):
            continue
        assigned = np.full(len(body), None, dtype=object)
        # The station array and the component mask are the same for every
        # section of a component, so they are built once per component rather
        # than once per section.
        for ck, g in eff_set.groupby("comp_key"):
            sta = station_of(body, g["sta_dir"].iloc[0])
            scope_mask = (np.ones(len(body), bool) if scope == "vehicle"
                          else (body["comp_key"] == ck).to_numpy())
            first = g["sta_lo"].min()
            for r in g.itertuples(index=False):
                lo = (sta >= r.sta_lo if abs(r.sta_lo - first) < 1e-9
                      else sta > r.sta_lo)
                sel = scope_mask & lo & (sta <= r.sta_hi)
                clash = sel & (assigned != None)          # noqa: E711
                if clash.any():
                    other = list(pd.unique(assigned[clash])[:3])
                    raise ValueError(f"config '{config}': {scope} section "
                                     f"'{r.section_id}' overlaps {other} on "
                                     f"{int(clash.sum())} node(s)")
                assigned[sel] = r.section_id
        body[col] = assigned
    return body


def _grouped_resultants(body, col, loadcases):
    """Sums F and the moment about the ORIGIN for each (loadcase, id) group.

    One pass over the body rather than a boolean scan per section per case.
    Moments are linear in the reference point, so the moment about any centroid
    follows from the origin sum and the force sum:

        M_c = M_0 - c x F
    """
    g = body[body[col].notna() & body["loadcase"].isin(loadcases)]
    if g.empty:
        return {}
    F = g[FCOLS].to_numpy(float)
    M0 = np.cross(g[XCOLS].to_numpy(float), F)
    acc = pd.DataFrame(
        {"loadcase": g["loadcase"].to_numpy(), "sid": g[col].to_numpy(),
         "fx": F[:, 0], "fy": F[:, 1], "fz": F[:, 2],
         "mx": M0[:, 0], "my": M0[:, 1], "mz": M0[:, 2], "n": 1})
    tot = acc.groupby(["loadcase", "sid"], sort=False).sum()
    return {k: (v[:3], v[3:6], int(v[6]))
            for k, v in zip(tot.index, tot.to_numpy(float))}


def section_loads(body, sections, config, loadcases, vehicle=None):
    """Six-component load at each section centroid, tagged by partition."""
    eff = effective_sections(sections, config)
    vkey = norm(vehicle) if vehicle else None
    sums = {c: _grouped_resultants(body, c, loadcases)
            for c in ("section_id", "vehicle_id") if c in body.columns}

    rows = []
    for r in eff.itertuples(index=False):
        cen = np.array([r.cen_x, r.cen_y, r.cen_z], float)
        is_veh = r.comp_key == vkey
        col = "vehicle_id" if is_veh else "section_id"
        for lc in loadcases:
            F, M0, n = sums.get(col, {}).get((lc, r.section_id),
                                             (np.zeros(3), np.zeros(3), 0))
            M = M0 - np.cross(cen, F)
            rows.append({"loadcase": lc, "partition": "vehicle" if is_veh
                         else "component", "configuration": r.configuration,
                         "component": r.component,
                         "section_id": r.section_id, "label": r.label,
                         "sta_dir": r.sta_dir, "lat_dir": r.lat_dir,
                         "sta_lo": r.sta_lo,
                         "sta_hi": r.sta_hi, "cen_x": cen[0], "cen_y": cen[1],
                         "cen_z": cen[2], "n_nodes": n,
                         "Fx": F[0], "Fy": F[1], "Fz": F[2],
                         "Mx": M[0], "My": M[1], "Mz": M[2]})
    return pd.DataFrame(rows)


def run_sections(df, sections, load_cases, vehicle=None):
    """Assigns and sums once per configuration. Returns (sec, {config: body})."""
    out, bodies = [], {}
    for cfg, g in load_cases.groupby("config"):
        lcs = sorted(g["loadcase"])
        body = assign_sections(df[df["loadcase"].isin(lcs)], sections, cfg,
                               vehicle)
        bodies[cfg] = body
        out.append(section_loads(body, sections, cfg, lcs, vehicle))

    sec = pd.concat(out, ignore_index=True)
    if sec.duplicated(["section_id", "loadcase"]).any():
        raise AssertionError("duplicate (section_id, loadcase) rows")
    empty = sorted(set(load_cases["loadcase"]) - set(sec["loadcase"]))
    if empty:
        raise ValueError(f"load case(s) {empty} resolved no sections")
    return sec.sort_values(["loadcase", "partition", "component", "sta_lo"]
                           ).reset_index(drop=True), bodies


def transfer(F, M, frm, to):
    """Moves a six-component load from `frm` to `to`."""
    F = np.asarray(F, float)
    return F, np.asarray(M, float) + np.cross(
        np.asarray(frm, float) - np.asarray(to, float), F)


def _sum_to(rows, cg):
    Ft, Mt = np.zeros(3), np.zeros(3)
    for r in rows.itertuples(index=False):
        F, M = transfer([r.Fx, r.Fy, r.Fz], [r.Mx, r.My, r.Mz],
                        [r.cen_x, r.cen_y, r.cen_z], cg)
        Ft += F
        Mt += M
    return np.concatenate([Ft, Mt])


def _pm_sum_to(pm_rows, cg):
    if not len(pm_rows):
        return np.zeros(6)
    F = pm_rows[FCOLS].to_numpy(float)
    X = pm_rows[XCOLS].to_numpy(float)
    return np.concatenate([F.sum(axis=0),
                           np.cross(X - np.asarray(cg, float), F).sum(axis=0)])


def node_report(df):
    """Node counts per tagged item."""
    per = df.drop_duplicates("node_id")
    out = (per.groupby("component")
           .agg(nodes=("node_id", "size"), excluded=("excluded", "sum"))
           .reset_index())
    out["in_sections"] = out["nodes"] - out["excluded"]
    return out.sort_values("component").reset_index(drop=True)


def assignment_report(bodies, sections, vehicle=None):
    """Sections, nodes assigned, orphans and gaps per configuration."""
    vkey = norm(vehicle) if vehicle else None
    rows = []
    for cfg, body in bodies.items():
        eff = effective_sections(sections, cfg)
        per = body.drop_duplicates("node_id")
        for subset, col, part in ((eff[eff["comp_key"] != vkey], "section_id",
                                   "component"),
                                  (eff[eff["comp_key"] == vkey], "vehicle_id",
                                   "vehicle")):
            for ck, g in subset.groupby("comp_key"):
                scope = per if part == "vehicle" else per[per["comp_key"] == ck]
                g = g.sort_values("sta_lo")
                gaps = (g["sta_lo"].to_numpy()[1:] - g["sta_hi"].to_numpy()[:-1])
                rows.append({"configuration": cfg, "partition": part,
                             "component": g["component"].iloc[0],
                             "sta_dir": g["sta_dir"].iloc[0],
                             "sections": len(g), "nodes": len(scope),
                             "assigned": int(scope[col].notna().sum()),
                             "orphans": int(scope[col].isna().sum()),
                             "gaps": int((np.abs(gaps) > 1e-9).sum())})
    return (pd.DataFrame(rows)
            .sort_values(["configuration", "partition", "component"])
            .reset_index(drop=True))


def check_partitions_agree(sec, cg, tol=1e-8):
    """Vehicle partition against the component partition, both at the CG."""
    if not (sec["partition"] == "vehicle").any():
        return {"ok": True, "worst": 0.0, "scale": 0.0, "err": pd.DataFrame()}
    cg = np.asarray(cg, float)
    rows = []
    for lc, g in sec.groupby("loadcase"):
        d = (_sum_to(g[g["partition"] == "vehicle"], cg)
             - _sum_to(g[g["partition"] == "component"], cg))
        rows.append({"loadcase": lc, **{f"d{k}": v for k, v in zip(SIX, d)}})
    err = pd.DataFrame(rows)
    scale = max(float(np.abs(sec[SIX].to_numpy()).max()), 1e-12)
    worst = float(err[[f"d{k}" for k in SIX]].abs().to_numpy().max())
    return {"ok": worst < tol * scale, "worst": worst, "scale": scale,
            "err": err}


def cg_breakdown(sec, pm_loads, cg, pm_label="point masses"):
    """Forces and moments about `cg`, section by section, with subtotals."""
    cg = np.asarray(cg, float)
    rows = []
    for lc, g in sec.groupby("loadcase"):
        pm = _pm_sum_to(pm_loads[pm_loads["loadcase"] == lc], cg)
        for part, gp in g.groupby("partition"):
            for comp, gc in gp.groupby("component"):
                for r in gc.sort_values("sta_lo").itertuples(index=False):
                    one = gc[gc["section_id"] == r.section_id]
                    rows.append({"loadcase": lc, "partition": part, "_grp": 0,
                                 "level": "section", "component": comp,
                                 "section_id": r.section_id, "label": r.label,
                                 "sta_lo": r.sta_lo, "sta_hi": r.sta_hi,
                                 "n_nodes": r.n_nodes,
                                 **dict(zip(SIX, _sum_to(one, cg)))})
                rows.append({"loadcase": lc, "partition": part, "_grp": 0,
                             "level": "subtotal", "component": comp,
                             "section_id": "", "label": f"{comp} total",
                             "sta_lo": np.nan, "sta_hi": np.nan,
                             "n_nodes": gc["n_nodes"].sum(),
                             **dict(zip(SIX, _sum_to(gc, cg)))})
            for level, grp, vals in ((pm_label, 1, pm),
                                     ("TOTAL", 2, _sum_to(gp, cg) + pm)):
                rows.append({"loadcase": lc, "partition": part, "_grp": grp,
                             "level": level, "component": "", "section_id": "",
                             "label": "", "sta_lo": np.nan, "sta_hi": np.nan,
                             "n_nodes": np.nan, **dict(zip(SIX, vals))})

    cols = ["loadcase", "partition", "level", "component", "section_id",
            "label", "sta_lo", "sta_hi", "n_nodes"] + SIX
    return (pd.DataFrame(rows)
            .sort_values(["loadcase", "partition", "_grp", "component",
                          "sta_lo"], na_position="last")
            .drop(columns="_grp")[cols].reset_index(drop=True))


def force_audit(df, pm_loads, cg, sec=None):
    """Card rows and point mass loads about `cg`. TOTAL is the equilibrium row."""
    cg = np.asarray(cg, float)

    def resultant(rows):
        if not len(rows):
            return np.zeros(6)
        F = rows[FCOLS].to_numpy(float)
        X = rows[XCOLS].to_numpy(float)
        return np.concatenate([F.sum(axis=0), np.cross(X - cg, F).sum(axis=0)])

    rows = []
    for lc, g in df.groupby("loadcase"):
        excl, used = g[g["excluded"]], g[~g["excluded"]]
        pm = _pm_sum_to(pm_loads[pm_loads["loadcase"] == lc], cg)
        entries = [("1 all card rows", resultant(g), len(g)),
                   ("2 held-out item rows", resultant(excl), len(excl)),
                   ("3 card rows in sections", resultant(used), len(used))]
        if sec is not None:
            comp = sec[(sec["loadcase"] == lc)
                       & (sec["partition"] == "component")]
            entries.append(("3a component sections", _sum_to(comp, cg),
                            len(comp)))
        entries += [("4 point mass loads", pm,
                     int((pm_loads["loadcase"] == lc).sum())),
                    ("5 TOTAL (3 + 4)", resultant(used) + pm, 0)]
        for name, vals, n in entries:
            rows.append({"loadcase": lc, "group": name, "rows": n,
                         **dict(zip(SIX, vals))})
    return pd.DataFrame(rows)


def section_nodes(bodies, loadcase, section_id=None, partition="component"):
    """The card rows behind a section, for diffing against a hand calc."""
    col = "vehicle_id" if partition == "vehicle" else "section_id"
    rows = []
    for body in bodies.values():
        g = body[body["loadcase"] == loadcase]
        if section_id is not None:
            g = g[g[col] == section_id]
        if len(g):
            rows.append(g)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    keep = ["node_id", "component", col] + XCOLS + FCOLS
    return out[[c for c in keep if c in out.columns]].sort_values(
        "node_id").reset_index(drop=True)


def export_section_loads(sec, path):
    """Writes loadcase, section, ids, centroid and the six components."""
    cols = ["loadcase", "partition", "component", "section_id", "label",
            "cen_x", "cen_y", "cen_z"] + SIX
    sec[cols].to_csv(path, index=False)
    return path


def export_point_mass_loads(pm_loads, path):
    """Writes point mass locations and forces across the load cases."""
    cols = ["loadcase", "node_id", "label", "mass"] + XCOLS + FCOLS
    cols = [c for c in cols if c in pm_loads.columns]
    pm_loads[cols].to_csv(path, index=False)
    return path

def station_diagram(sec, partition="component", direction="low"):
    """Cumulative shear, axial and bending along each component's stations.

    Sections are disjoint bins, so a running sum gives the classic diagram. Each
    section is transferred from its centroid to the cut; the cut lies on a datum
    through the mean of that component's centroids with the station coordinate
    swept.

    Args:
        partition: "component" (default), "vehicle", or None for both. The
            vehicle partition spans every node and covers the same load as the
            component partition, so it is left out by default rather than
            plotted alongside.
        direction: which end to accumulate from. "low" starts at the lowest
            station, "high" at the highest. Accepts a dict keyed by component
            name for a mixed model -- a fuselage usually accumulates from the
            nose ("low"), while a wing accumulates from the tip, which is "high"
            for a RH wing on positive y and "low" for a LH wing on negative y.

    Returns:
        loadcase, component, station, direction and the six cumulative values.
    """
    use = sec if partition is None else sec[sec["partition"] == partition]
    rows = []
    for (part, lc, cfg, comp), g in use.groupby(
            ["partition", "loadcase", "configuration", "component"]):
        how = (direction.get(comp, "low") if isinstance(direction, dict)
               else (direction or "low"))
        if how not in ("low", "high"):
            raise ValueError(f"direction must be 'low' or 'high', got '{how}'")
        axis = parse_sta_dir(g["sta_dir"].iloc[0])
        datum = {c: float(g[f"cen_{c}"].mean()) for c in XCOLS if c != axis}
        # Anchor the free end at zero so the diagram starts from nothing.
        edge = "sta_hi" if how == "low" else "sta_lo"
        anchor = (g["sta_lo"].min() if how == "low" else g["sta_hi"].max())
        cuts = np.concatenate([[anchor], g[edge].to_numpy(float)])
        for cut in cuts:
            upto = (g[g["sta_hi"] <= cut + 1e-9] if how == "low"
                    else g[g["sta_lo"] >= cut - 1e-9])
            ref = np.array([cut if c == axis else datum[c] for c in XCOLS],
                           float)
            rows.append({"loadcase": lc, "partition": part,
                         "configuration": cfg,
                         "component": comp, "sta_dir": axis,
                         "lat_dir": g["lat_dir"].iloc[0], "direction": how,
                         "station": cut, "sections": len(upto),
                         **dict(zip(SIX, _sum_to(upto, ref)))})
    return (pd.DataFrame(rows)
            .sort_values(["partition", "component", "configuration",
                          "loadcase", "station"])
            .reset_index(drop=True))


def station_drivers(diag, quantities=None):
    """Peak magnitude per component and configuration, with the case that sets it."""
    rows = []
    for (part, comp, cfg), g in diag.groupby(
            ["partition", "component", "configuration"]):
        pairs = station_columns(g["sta_dir"].iloc[0], g["lat_dir"].iloc[0])
        if quantities:
            roles = dict(pairs)
            pairs = [(q, roles.get(q, "")) for q in quantities]
        for q, role in pairs:
            i = g[q].abs().idxmax()
            rows.append({"partition": part, "component": comp,
                         "configuration": cfg,
                         "quantity": q, "axis": role, "peak": g.loc[i, q],
                         "station": g.loc[i, "station"],
                         "loadcase": g.loc[i, "loadcase"]})
    return pd.DataFrame(rows)


# sta_dir is normal to the section plane; lat_dir is the first lateral axis and
# the second is what is left. All six components are global.
_FORCE = {"x": "Fx", "y": "Fy", "z": "Fz"}
_MOMENT = {"x": "Mx", "y": "My", "z": "Mz"}
ROLES = ["normal", "lateral 1", "lateral 2"]


def station_axes(sta_dir, lat_dir):
    """Axis letters for normal, lateral 1 and lateral 2."""
    nrm, lat1 = parse_sta_dir(sta_dir), parse_sta_dir(lat_dir)
    if lat1 == nrm:
        raise ValueError(f"lat_dir '{lat1}' cannot be the station axis '{nrm}'")
    lat2 = next(c for c in XCOLS if c not in (nrm, lat1))
    return dict(zip(ROLES, (nrm, lat1, lat2)))


def station_columns(sta_dir, lat_dir):
    """Ordered (column, role) pairs: forces then moments, normal first."""
    ax = station_axes(sta_dir, lat_dir)
    return ([(_FORCE[ax[r]], r) for r in ROLES]
            + [(_MOMENT[ax[r]], r) for r in ROLES])


def panel_drivers(g, column, n=None):
    """Cases that set the highest or lowest value at any station in a panel.

    These are exactly the cases that touch the envelope. `n` caps the list,
    keeping the ones that are extreme at the most stations; None returns all.
    """
    hi = g.loc[g.groupby("station")[column].idxmax(), "loadcase"]
    lo = g.loc[g.groupby("station")[column].idxmin(), "loadcase"]
    order = pd.concat([hi, lo]).value_counts().index.tolist()
    return order if n is None else order[:n]


def plot_station_diagram(diag, component, configuration=None, out_dir=None,
                         highlight=None, drivers=None, envelope=True,
                         figsize=(15, 7)):
    """Six panels -- forces then moments, normal then lateral 1 then lateral 2.

    With many load cases a per-case legend is unreadable, so every case is drawn
    faint, the min/max envelope is shaded, and only the cases that set an
    extreme are coloured and named.

    Args:
        highlight: cases to colour regardless. Overrides the automatic pick.
        drivers: cap on how many envelope-touching cases to colour. None
            colours every case that is a maximum or minimum at any station,
            which is the full set that defines the envelope.
        envelope: shade between the minimum and maximum across cases.
    """
    import matplotlib.pyplot as plt

    g = diag[diag["component"] == component]
    if configuration is not None:
        g = g[g["configuration"] == configuration]
    if g.empty:
        raise ValueError(f"no rows for '{component}' / '{configuration}'")
    if g["configuration"].nunique() > 1:
        raise ValueError(
            f"'{component}' has configurations "
            f"{sorted(g['configuration'].unique())}; pass configuration= to "
            "plot one at a time")

    pairs = station_columns(g["sta_dir"].iloc[0], g["lat_dir"].iloc[0])
    cases = sorted(g["loadcase"].unique())
    forced = None if highlight is None else [str(h) for h in highlight]

    # A panel whose values are negligible against the others is round-off, not
    # load. Marked so an axis reading 1e-13 is not mistaken for a result.
    biggest = max(float(np.abs(g[q].to_numpy(float)).max()) for q, _ in pairs)
    fig, axes = plt.subplots(2, 3, figsize=figsize, sharex=True)
    for ax, (q, role) in zip(axes.ravel(), pairs):
        scale = float(np.abs(g[q].to_numpy(float)).max())
        negligible = scale < 1e-9 * max(biggest, 1e-30)
        band = g.groupby("station")[q].agg(["min", "max"]).sort_index()
        if envelope and len(cases) > 1:
            ax.fill_between(band.index, band["min"], band["max"],
                            color="0.85", zorder=0)
        for lc in cases:
            d = g[g["loadcase"] == lc].sort_values("station")
            ax.plot(d["station"], d[q], color="0.55", lw=0.5, alpha=0.5,
                    zorder=1)
        lead = forced if forced is not None else panel_drivers(g, q, drivers)
        for lc in lead:
            d = g[g["loadcase"] == lc].sort_values("station")
            ax.plot(d["station"], d[q], lw=1.8, marker="o", markersize=3,
                    label=lc, zorder=3)
        ax.axhline(0, color="0.4", lw=0.8, zorder=2)
        ax.set_title(f"{q}   {role}" + ("   (~0)" if negligible else ""),
                     fontsize=10)
        ax.grid(alpha=0.25)
        if negligible:
            ax.set_ylim(-1, 1)
            ax.text(0.5, 0.5, "negligible", transform=ax.transAxes,
                    ha="center", va="center", color="0.5", fontsize=9)
        elif len(lead) > 1 or forced is not None:
            ax.legend(fontsize=6, loc="best", framealpha=0.85,
                      ncol=max(1, min(4, (len(lead) + 7) // 8)))
    for ax in axes[-1]:
        ax.set_xlabel("station")
    cfg = g["configuration"].iloc[0]
    note = (f"{len(cases)} load cases; coloured lines touch the envelope"
            if forced is None else f"{len(cases)} load cases, highlighted: "
            + ", ".join(forced))
    fig.suptitle(f"{component} — {cfg}\n{note}", fontsize=11)
    fig.tight_layout()
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{norm(component)}_{norm(cfg)}".replace(" ", "_")
        fig.savefig(out / f"stations_{stem}.png", dpi=140)
    return fig


def plot_all_station_diagrams(diag, out_dir=None, highlight=None, **kw):
    """One figure per component and configuration present in the diagram."""
    figs = {}
    for comp, cfg in (diag[["component", "configuration"]]
                      .drop_duplicates().itertuples(index=False)):
        figs[(comp, cfg)] = plot_station_diagram(
            diag, comp, cfg, out_dir=out_dir, highlight=highlight, **kw)
    return figs


def configuration_report(df):
    """Per configuration and load case: rows, unique nodes, and whether the
    node set is the same across cases."""
    rows = []
    for cfg, g in df.groupby("config_folder"):
        per = g.groupby("loadcase")["node_id"].agg(["size", "nunique"])
        usual = int(per["nunique"].mode().iloc[0])
        for lc, r in per.iterrows():
            rows.append({"configuration": cfg, "loadcase": lc,
                         "rows": int(r["size"]), "nodes": int(r["nunique"]),
                         "vs_usual": int(r["nunique"]) - usual,
                         "repeats": int(r["size"] - r["nunique"])})
    return pd.DataFrame(rows).sort_values(["configuration", "loadcase"]
                                          ).reset_index(drop=True)


def scale_report(df, pm_loads):
    """Card loads against point mass loads, per case.

    Totals of absolute components, not maxima. A single point mass will exceed
    a single grid point force by orders of magnitude, so only the totals are
    comparable.
    """
    rows = []
    for lc, g in df.groupby("loadcase"):
        F = g[FCOLS].to_numpy(float)
        p = pm_loads[pm_loads["loadcase"] == lc][FCOLS].to_numpy(float)
        card_abs = float(np.abs(F).sum())
        pm_abs = float(np.abs(p).sum()) if len(p) else 0.0
        rows.append({"loadcase": lc, "nodes": g["node_id"].nunique(),
                     "card_sum_abs": card_abs,
                     "card_resultant": float(np.linalg.norm(F.sum(axis=0))),
                     "card_max_node": float(np.abs(F).max()) if len(F) else 0.0,
                     "pm_count": len(p), "pm_sum_abs": pm_abs,
                     "pm_max": float(np.abs(p).max()) if len(p) else 0.0,
                     "pm/card_totals": pm_abs / card_abs if card_abs else np.nan})
    return pd.DataFrame(rows)


def apply_exclusions(df, load_cases):
    """Removes each case's exclude_nodes rows from the force cards.

    Point masses are dropped by point_mass_loads from the same column, so after
    this one declaration has removed a node from both paths.

    Returns:
        (df without those rows, a per-case report of what was removed).
    """
    if "exclude_nodes" not in load_cases.columns:
        return df, pd.DataFrame()
    drops = load_cases.set_index("loadcase")["exclude_nodes"].to_dict()
    keep = np.ones(len(df), bool)
    rows = []
    for lc, g in df.groupby("loadcase"):
        drop = drops.get(lc) or set()
        hit = g["node_id"].isin(drop)
        keep[g.index[hit]] = False
        rows.append({"loadcase": lc, "declared": len(drop),
                     "rows_removed": int(hit.sum()),
                     "nodes_removed": int(g.loc[hit, "node_id"].nunique()),
                     "not_present": len(drop - set(g["node_id"]))})
    return df[keep].reset_index(drop=True), pd.DataFrame(rows)