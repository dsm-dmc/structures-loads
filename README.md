# Section loads

Turns nodal force card files into a six-component load (Fx/Fy/Fz, Mx/My/Mz) at
each section centroid, for every load case, plus the point mass inertia loads.

---

## Just want to run it?

You do not need to know anything about Git or Python.

1. On this page, click the green **Code** button near the top right.
2. Click **Download ZIP**.
3. Unzip it somewhere on your computer, like your Desktop. Not inside the ZIP
   itself -- right-click the file and choose "Extract All".
4. Open the unzipped folder and double-click **run_dashboard.bat**.

A black console window opens, then the dashboard opens in your browser. The
first time takes a few minutes while it sets itself up. After that it starts in
seconds.

To stop it, close the black console window.

**If it says Python was not found:** install Python from
[python.org/downloads](https://www.python.org/downloads/). During install, tick
"Add python.exe to PATH". You do not need administrator rights. Then
double-click `run_dashboard.bat` again.

Nothing is installed outside the folder you unzipped. Delete the folder and it
is gone.

---

## Using the dashboard

Six tabs across the top:

| tab | what it shows |
|---|---|
| Inputs | the load files and the four input tables, read only |
| Checks | whether anything is missing or does not add up |
| Section loads | the results table, with CSV downloads |
| Plots | loads along the stations, and which load cases drive them |
| Breakdown | forces and moments about the vehicle CG |
| Tests | press a button to check the code and your data |

The sidebar on the left points at the data folder and holds the vehicle CG and
unit settings. Change something, press **Run**.

To change the inputs, edit the CSV files in the `data` folder in Excel and press
**Run**. The dashboard does not edit them.

---

## For developers

```bash
poetry install
poetry run pytest -q                        # 158 tests
poetry run python scripts/run_pipeline.py   # command line, end to end
poetry run jupyter lab                      # notebook, see scripts/notebook_cells.py
```

`docs/METHOD.md` explains how the loads are derived. `docs/TESTING.md` covers
the test suite.

### Inputs

| file | what it defines |
|---|---|
| `data/*_Force_Cards/*.txt` | nodal forces, one file per load case |
| `data/sections.csv` | where each section starts and ends, and its centroid |
| `data/node_ranges.csv` | which node ID ranges belong to which component |
| `data/load_cases.csv` | load factors per case, and any expected missing nodes |
| `data/point_masses.csv` | mass and location of each point mass |

Force card filenames need an `LC###` token. The load case comes from that token
and the configuration from the folder name.

### Outputs

```
outputs/section_loads.csv       the deliverable
outputs/point_mass_loads.csv    point mass forces per case
outputs/station_diagram.csv     cumulative values along the stations
outputs/images/                 one plot per component and configuration
```

The first run of `run_dashboard.bat` needs internet for pip. If a machine has
none, run `pip download -r requirements.txt -d wheels` where you do have
internet and ship the `wheels` folder alongside; the script uses it
automatically.