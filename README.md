# Section loads

Reads nodal force card files and produces a six-component load (Fx/Fy/Fz,
Mx/My/Mz) at each section centroid, for every load case, plus the point mass
inertia loads.

## First-time setup

Install Poetry if you don't have it:

```bash
pipx install poetry
```

Then from the project root:

```bash
poetry install
```

That creates a virtual environment and installs numpy, pandas, pytest and
Jupyter. Check it worked:

```bash
poetry run pytest -q
```

You should see `58 passed`. That test run doesn't touch your data, so it
passing means the code itself is fine.

## Running it

Command line, end to end:

```bash
poetry run python scripts/run_pipeline.py
```

Or in a notebook:

```bash
poetry run jupyter lab
```

Create a notebook in `scripts/` and copy the cells from
`scripts/notebook_cells.py` into it, one block per cell. Each cell ends on a
dataframe, so results display as tables.

## What goes where

```
data/
    CC_Force_Cards/*.txt     force cards for captive carry cases
    FF_Force_Cards/*.txt     force cards for free flight cases
    sections.csv             section definitions
    node_ranges.csv          which node IDs belong to which component
    load_cases.csv           load factors per case
    point_masses.csv         point masses and their locations
outputs/
    section_loads.csv        the deliverable
    point_mass_loads.csv     point mass forces per case
```

Force card filenames need an `LC###` token, e.g. `anything_LC101_anything.txt`.
The load case comes from that token and the configuration from the folder name.

## Settings

The top of `scripts/notebook_cells.py`:

```python
VEHICLE = "Vehicle"                    # component whose sections span everything
EXCLUDE_ITEMS = {"Point Masses"}       # node_ranges items kept out of the sections
VEHICLE_CG = np.array([52.0, 0.0, -0.5])
INERTIA_SIGN = -1.0                    # point mass force = -mass * N
ACCEL_TO_G = 1.0                       # 386.09 if nx/ny/nz are in/s^2
ONLY_LOADCASES = []                    # e.g. ["LC101"] while iterating
```

Set `VEHICLE_CG` to your vehicle's centre of gravity. Leave the rest alone
unless you know you need to change them.

## Reading the output

Each notebook cell shows a table:

| cell | table | what to look for |
|---|---|---|
| 2 | load cases | the cases and configurations that were found |
| 3 | validation | `agree` should be True; the second table should be empty |
| 4 | nodes | node counts per component, none `UNTAGGED` |
| 5 | assignment | `orphans` and `gaps` should be 0 |
| 6 | force audit | where every card row went |
| 7 | breakdown | forces and moments about the CG, section by section |

Anything genuinely wrong raises an error with a message naming the file and
rows involved, rather than producing quiet nonsense.
