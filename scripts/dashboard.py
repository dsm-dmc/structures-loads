"""Streamlit dashboard.

Three steps, so nothing slow happens until you ask for it:
    1  read the four tables
    2  read the load files
    3  run the loads and plot

    poetry run streamlit run scripts/dashboard.py
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from section_loads import (  # noqa: E402
    apply_exclusions, assignment_report, cg_breakdown,
    check_card_configurations, check_component_coverage, check_configurations,
    check_partitions_agree, check_sections_per_configuration, force_audit,
    node_report, norm, plot_station_diagram, point_mass_loads, read_force_cards,
    read_tables, run_sections, station_diagram, station_drivers,
    tag_components)

st.set_page_config(page_title="Section loads", layout="wide")
S = st.session_state


def clear(*keys):
    for k in keys:
        S.pop(k, None)


st.title("Section loads")

with st.sidebar:
    st.header("Data")
    data_dir = st.text_input("Folder", str(ROOT / "data"))
    if st.button("Reload tables", use_container_width=True):
        clear("tables", "cards", "results")

if "tables" not in S or S.get("data_dir") != data_dir:
    try:
        S.tables = read_tables(data_dir)
        S.data_dir = data_dir
        clear("cards", "results")
    except Exception as e:                                 # noqa: BLE001
        st.error(f"read_tables: {type(e).__name__}: {e}")
        st.stop()
sections, node_ranges, load_cases_all, point_masses = S.tables

components = sorted(sections["component"].unique())
items = sorted(node_ranges["item"].unique())

with st.sidebar:
    st.header("Settings")
    vehicle = st.selectbox("Vehicle component", components,
                           index=next((i for i, c in enumerate(components)
                                       if norm(c) == "vehicle"), 0))
    exclude = set(st.multiselect("Items kept out of sections", items,
                                 default=[i for i in items
                                          if "mass" in norm(i)]))
    from_high = set(st.multiselect(
        "Accumulate from the high station end", components,
        default=[c for c in components if norm(c).endswith("lh")],
        help="A fuselage runs from the nose. A wing runs from the tip: high "
             "on positive y, low on negative y."))
    cg_txt = st.text_input("Vehicle CG  x, y, z", "149.0, 0.0, 100")
    inertia_sign = st.selectbox("Inertia sign", [-1.0, 1.0])
    accel_to_g = st.number_input("accel_to_g", value=1.0, format="%.4f",
                                 help="386.09 only if nx/ny/nz are in in/s^2")
    only = st.multiselect("Only these load cases (blank = all)",
                          sorted(load_cases_all["loadcase"]))

try:
    cg = np.array([float(v) for v in cg_txt.split(",")])
    assert cg.shape == (3,)
except Exception:                                          # noqa: BLE001
    st.error("Vehicle CG must be three numbers, e.g. 52.0, 0.0, -0.5")
    st.stop()
station_from = {c: "high" for c in from_high}

# ------------------------------------------------------------- 1. inputs -----
st.subheader("1 — inputs")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Load cases", len(load_cases_all))
c2.metric("Sections", len(sections))
c3.metric("Node ranges", len(node_ranges))
c4.metric("Point masses", len(point_masses))

cards_dir = Path(data_dir)
files = sorted(p.name for p in cards_dir.rglob("*.txt"))
st.write(f"**Load input 1, nodal force cards:** {len(files)} file(s) under "
         f"`{cards_dir}`")
st.write(f"**Load input 2, point mass body loads:** {len(point_masses)} "
         f"mass(es), {point_masses['mass'].sum():,.2f} total")

with st.expander("Tables as read"):
    st.caption("Read only. Edit the CSV, then press Reload in the sidebar.")
    for name, frame in (("load_cases.csv", load_cases_all),
                        ("sections.csv", sections),
                        ("node_ranges.csv", node_ranges),
                        ("point_masses.csv", point_masses)):
        st.markdown(f"**{name}**")
        st.dataframe(frame, use_container_width=True, hide_index=True)

# --------------------------------------------------------- 2. read cards -----
st.subheader("2 — read the load files")
if st.button("Read load files", type="primary"):
    clear("results")
    with st.spinner(f"reading {len(files)} file(s)..."):
        try:
            df, load_cases = read_force_cards(data_dir, load_cases_all,
                                              only=only or None)
            df, excluded = apply_exclusions(df, load_cases)
            S.cards = dict(df=df, load_cases=load_cases, excluded=excluded)
        except Exception as e:                             # noqa: BLE001
            st.error(f"{type(e).__name__}: {e}")

if "cards" not in S:
    st.info("Not read yet.")
    st.stop()

card = S.cards
st.write(f"{len(card['df']):,} rows, "
         f"{card['df']['node_id'].nunique():,} unique nodes, "
         f"{card['load_cases']['loadcase'].nunique()} load case(s)")
with st.expander("Files and exclusions"):
    st.dataframe(card["df"].drop_duplicates("loadcase")[
        ["loadcase", "config_folder", "source_file"]],
        use_container_width=True, hide_index=True)
    if len(card["excluded"]):
        st.dataframe(card["excluded"], use_container_width=True,
                     hide_index=True)

# ---------------------------------------------------------------- 3. run -----
st.subheader("3 — run the loads")
if st.button("Run", type="primary"):
    with st.spinner("binning sections..."):
        try:
            veh, excl = vehicle, exclude
            check_component_coverage(sections, node_ranges, veh, excl)
            check_configurations(sections, load_cases_all)
            df = tag_components(card["df"], node_ranges, excl)
            pm = point_mass_loads(point_masses, card["load_cases"],
                                  inertia_sign, accel_to_g)
            sec, bodies = run_sections(df, sections, card["load_cases"], veh)
            diag = station_diagram(sec, direction=station_from or "low")
            S.results = dict(
                sec=sec, pm=pm, diag=diag,
                nodes=node_report(df),
                assign=assignment_report(bodies, sections, veh),
                cards=check_card_configurations(df, card["load_cases"]),
                gaps=check_sections_per_configuration(
                    sections, node_ranges, load_cases_all, veh, excl),
                agree=check_partitions_agree(sec, cg),
                audit=force_audit(df, pm, cg, sec),
                breakdown=cg_breakdown(sec, pm, cg),
                drivers=station_drivers(diag))
        except Exception as e:                             # noqa: BLE001
            st.error(f"{type(e).__name__}: {e}")

if "results" not in S:
    st.info("Not run yet.")
    st.stop()

r = S.results
tabs = st.tabs(["Checks", "Section loads", "Plots", "Breakdown", "Tests"])

with tabs[0]:
    bad = r["assign"][(r["assign"]["orphans"] > 0) | (r["assign"]["gaps"] > 0)]
    c1, c2 = st.columns(2)
    c1.metric("Partitions agree", "yes" if r["agree"]["ok"] else "NO")
    c2.metric("Orphans or gaps", "none" if bad.empty else f"{len(bad)} row(s)")
    if len(r["gaps"]):
        st.warning("Items with no sections in some configuration.")
    for name, frame in (("Nodes", r["nodes"]), ("Assignment", r["assign"]),
                        ("Card configuration", r["cards"]),
                        ("Force audit", r["audit"])):
        st.markdown(f"**{name}**")
        st.dataframe(frame, use_container_width=True, hide_index=True)

with tabs[1]:
    st.dataframe(r["sec"], use_container_width=True, hide_index=True)
    st.download_button("section_loads.csv",
                       r["sec"].to_csv(index=False).encode(),
                       "section_loads.csv")
    st.download_button("point_mass_loads.csv",
                       r["pm"].to_csv(index=False).encode(),
                       "point_mass_loads.csv")
    st.markdown("**Point mass loads**")
    st.dataframe(r["pm"], use_container_width=True, hide_index=True)

with tabs[2]:
    diag = r["diag"]
    pairs = list(diag[["component", "configuration"]].drop_duplicates()
                 .itertuples(index=False))
    st.caption("Every case is drawn faint with the envelope shaded. Coloured "
               "lines are the cases that set a maximum or minimum somewhere.")
    for comp, conf in pairs:
        st.markdown(f"**{comp} — {conf}**")
        st.pyplot(plot_station_diagram(diag, comp, conf))
    st.markdown("**Drivers**")
    st.dataframe(r["drivers"], use_container_width=True, hide_index=True)

with tabs[3]:
    st.dataframe(r["breakdown"], use_container_width=True, hide_index=True)

with tabs[4]:
    st.write(f"Runs pytest against the code and `{data_dir}`.")
    if st.button("Run tests"):
        out = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header",
             "--tb=short", "-p", "no:cacheprovider"],
            cwd=ROOT, capture_output=True, text=True,
            env={**os.environ, "SECTION_LOADS_DATA": str(data_dir)})
        (st.success if out.returncode == 0 else st.error)(
            "passed" if out.returncode == 0 else "failures")
        st.code(out.stdout[-20000:] or out.stderr[-20000:])