"""Streamlit dashboard. Launch with run_dashboard.bat, or:

    poetry run streamlit run scripts/dashboard.py
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from section_loads import (  # noqa: E402
    apply_exclusions, assignment_report, check_card_configurations,
    check_component_coverage, check_configurations, check_partitions_agree,
    check_sections_per_configuration, cg_breakdown, export_point_mass_loads,
    export_section_loads, force_audit, node_report, norm, plot_station_diagram,
    point_mass_loads, read_force_cards, read_tables, run_sections,
    station_diagram, station_drivers)

st.set_page_config(page_title="Section loads", layout="wide")


@st.cache_data(show_spinner=False)
def run(data_dir, vehicle, exclude, only, inertia_sign, accel_to_g, cg, froms):
    froms = {c: "high" for c in (froms or ())}
    sections, node_ranges, load_cases, point_masses = read_tables(data_dir)
    df, load_cases = read_force_cards(data_dir, load_cases, only=only or None)
    df, excluded = apply_exclusions(df, load_cases)

    check_component_coverage(sections, node_ranges, vehicle, exclude)
    check_configurations(sections, load_cases)
    cards = check_card_configurations(df, load_cases)
    gaps = check_sections_per_configuration(sections, node_ranges, load_cases,
                                            vehicle, exclude)

    df = tag(df, node_ranges, exclude)
    pm = point_mass_loads(point_masses, load_cases, inertia_sign, accel_to_g)
    sec, bodies = run_sections(df, sections, load_cases, vehicle)
    diag = station_diagram(sec, direction=froms or "low")
    return dict(sections=sections, node_ranges=node_ranges,
                point_masses=point_masses,
                load_cases=load_cases, df=df, pm=pm,
                sec=sec, bodies=bodies, diag=diag, excluded=excluded,
                cards=cards, gaps=gaps,
                nodes=node_report(df),
                assign=assignment_report(bodies, sections, vehicle),
                agree=check_partitions_agree(sec, cg),
                audit=force_audit(df, pm, cg, sec),
                breakdown=cg_breakdown(sec, pm, cg),
                drivers=station_drivers(diag))


def tag(df, node_ranges, exclude):
    from section_loads import tag_components
    return tag_components(df, node_ranges, exclude)


st.title("Section loads")

with st.sidebar:
    st.header("Inputs")
    data_dir = st.text_input("Data folder", str(ROOT / "data"))
    vehicle = st.text_input("Vehicle component", "Vehicle")
    exclude_txt = st.text_input("Items excluded from sections", "Point Masses")
    only_txt = st.text_input("Only these load cases (blank = all)", "")
    st.header("Settings")
    from_high = st.text_input("Accumulate from the high station for", "RH Wing")
    st.caption("Comma separated. A wing runs from the tip: high on +y, low on -y.")
    cg_txt = st.text_input("Vehicle CG  x,y,z", "52.0, 0.0, -0.5")
    inertia_sign = st.selectbox("Inertia sign", [-1.0, 1.0])
    accel_to_g = st.number_input("accel_to_g", value=1.0, format="%.4f")
    st.caption("386.09 only if nx/ny/nz are in in/s^2")
    go = st.button("Run", type="primary", use_container_width=True)

exclude = {s.strip() for s in exclude_txt.split(",") if s.strip()}
only = [s.strip() for s in only_txt.split(",") if s.strip()]
cg = np.array([float(v) for v in cg_txt.split(",")])
froms = {c.strip(): "high" for c in from_high.split(",") if c.strip()}

if go:
    st.cache_data.clear()

try:
    r = run(data_dir, vehicle, frozenset(exclude), tuple(only), inertia_sign,
            accel_to_g, tuple(cg), tuple(sorted(froms)))
except Exception as e:                                    # noqa: BLE001
    st.error(f"{type(e).__name__}: {e}")
    st.stop()

tabs = st.tabs(["Inputs", "Checks", "Section loads", "Plots", "Breakdown",
                "Tests"])

with tabs[0]:
    st.subheader("Load input 1 — nodal force cards")
    src = r["df"].drop_duplicates("loadcase")[
        ["loadcase", "config_folder", "source_file"]]
    st.write(f"{len(src)} file(s), {len(r['df']):,} rows, "
             f"{r['df']['node_id'].nunique():,} unique nodes")
    st.dataframe(src, use_container_width=True, hide_index=True)

    st.subheader("Load input 2 — point mass body loads")
    st.write("Force is inertia_sign x mass x load factor, applied at each mass.")
    st.dataframe(r["point_masses"], use_container_width=True, hide_index=True)

    st.subheader("Tables")
    st.caption("Read only. Edit the CSV and press Run to pick up changes.")
    for name, frame in (("load_cases.csv", r["load_cases"]),
                        ("sections.csv", r["sections"]),
                        ("node_ranges.csv", r["node_ranges"])):
        st.markdown(f"**{name}**")
        st.dataframe(frame, use_container_width=True, hide_index=True)

with tabs[1]:
    a = r["agree"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Load cases", r["load_cases"]["loadcase"].nunique())
    c2.metric("Nodes", r["df"]["node_id"].nunique())
    c3.metric("Partitions agree", "yes" if a["ok"] else "NO")
    bad = r["assign"][(r["assign"]["orphans"] > 0) | (r["assign"]["gaps"] > 0)]
    if len(bad):
        st.warning("Sections do not cover every node.")
    if len(r["gaps"]):
        st.warning("Items with no sections in some configuration.")
    st.subheader("Nodes")
    st.dataframe(r["nodes"], use_container_width=True, hide_index=True)
    st.subheader("Assignment")
    st.dataframe(r["assign"], use_container_width=True, hide_index=True)
    st.subheader("Card configuration")
    st.dataframe(r["cards"], use_container_width=True, hide_index=True)
    if len(r["excluded"]):
        st.subheader("Exclusions applied")
        st.dataframe(r["excluded"], use_container_width=True, hide_index=True)
    st.subheader("Force audit")
    st.dataframe(r["audit"], use_container_width=True, hide_index=True)

with tabs[2]:
    st.dataframe(r["sec"], use_container_width=True, hide_index=True)
    st.download_button("section_loads.csv",
                       r["sec"].to_csv(index=False).encode(),
                       "section_loads.csv", use_container_width=True)
    st.download_button("point_mass_loads.csv",
                       r["pm"].to_csv(index=False).encode(),
                       "point_mass_loads.csv", use_container_width=True)

with tabs[3]:
    diag = r["diag"]
    pairs = diag[["component", "configuration"]].drop_duplicates()
    labels = [f"{c} - {g}" for c, g in pairs.itertuples(index=False)]
    pick = st.selectbox("Component", labels)
    comp, cfg = pick.rsplit(" - ", 1)
    st.pyplot(plot_station_diagram(diag, comp, cfg))
    st.subheader("Drivers")
    st.dataframe(r["drivers"], use_container_width=True, hide_index=True)

with tabs[4]:
    st.dataframe(r["breakdown"], use_container_width=True, hide_index=True)

with tabs[5]:
    st.write("Runs pytest against the code and the data folder above.")
    if st.button("Run tests"):
        env = {"SECTION_LOADS_DATA": str(data_dir)}
        out = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header",
             "--tb=short", "-p", "no:cacheprovider"],
            cwd=ROOT, capture_output=True, text=True,
            env={**dict(__import__("os").environ), **env})
        (st.success if out.returncode == 0 else st.error)(
            "passed" if out.returncode == 0 else "failures")
        st.code(out.stdout[-20000:] or out.stderr[-20000:])