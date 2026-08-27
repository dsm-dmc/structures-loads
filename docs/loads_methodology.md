# How the loads are derived

Two things the structures team applies to the FEM:

1. **Section loads** - Structures RBE3s from that centroid to the surrounding structure.
2. **Point mass loads** - one force per point mass per load case, applied at the mass location.

These are separate and are not summed. Point mass nodes are held out of the section sums.

## Inputs

Customer-supplied nodal force cards, one file per load case:

```
node_id, x, y, z, fx, fy, fz
```

These contain the total applied load at each grid point - aero and inertia. Plus four tables you maintain, in `data/`:

| file | what it defines |
|---|---|
| `sections.csv` | where each section starts and ends, and its centroid |
| `node_ranges.csv` | which node ID ranges belong to which component |
| `load_cases.csv` | load factors nx, ny, nz per case |
| `point_masses.csv` | mass and location of each point mass |

## Body loads (point masses)

Each point mass gets a force per load case:

```
F = -mass * N
```

`N` is the load factor from `load_cases.csv`. Mass is in lbm and `N` in g, so the result is lbf directly - there is no 386.09 factor. The minus sign is inertia opposing the acceleration.

These are applied as FORCE cards at the mass locations.

## Section loads

**1. Tag every node to a component.** Node IDs are matched against the ranges in `node_ranges.csv`. Anything matching no range is reported as `UNTAGGED` and isn't processed.

**2. Bin nodes into sections.** Each section has a station direction (`sta_dir`, one of x/y/z) and a start and end station. A node's station is its
raw global coordinate along that axis, so a left-hand component has negative brackets. Bins are `(start, end]` - a node on a shared boundary belongs to the lower section. Only the lowest section of a component includes its bottom boundary.

**3. Sum each bin about its centroid.**

```
F = sum of fx, fy, fz over the nodes in the section
M = sum of (r_node - r_centroid) x F_node
```

In global axes.

## Configurations

Sections and load cases are tagged `ALL`, `FF` or `CC`. A component can't be `ALL` and also have a configuration. 

## Two partitions

Sections are grouped two ways, over the same nodes:

- **component** - fuselage sections bin fuselage nodes, wing sections bin wing   nodes, and so on.
- **vehicle** - sections that bin *every* node regardless of component, giving  whole-vehicle station bands.

They cover the same load and are never summed together. Both totals must agree at the CG, which is a cross-check on both.

## Station diagrams

Sections may be disjoint bins, so a running sum from one end gives the classic shear and bending diagram. `STATION_FROM` sets which end each component
starts from

The plots show six things: forces then moments, in the order normal, lateral 1, lateral 2. `sta_dir` is the normal (perpendicular to the section plane),
`lat_dir` is lateral 1, and lateral 2 is whichever axis is left.

`station_drivers` lists the peak of each component and the load case that sets it. A case appearing nowhere in that table is not sizing anything.

## Running it

```bash
poetry install
poetry run pytest -q                        # 80 passed, does not touch data
poetry run python scripts/run_pipeline.py   # end to end
```

Or in a notebook, copying the blocks from `scripts/notebook_cells.py`:

| cell | does |
|---|---|
| 1 | settings |
| 2 | read the cards and tables |
| 3 | validation |
| 4 | tag nodes to components |
| 5 | point mass loads and section loads |
| 6 | force audit |
| 7 | forces and moments about the CG |
| 8 | station diagrams |
| 9 | plots to `outputs/images/` |
| 10 | export CSVs |

## Adding load cases

1. Drop the force card file in `data/CC_Force_Cards/` or `data/FF_Force_Cards/`. The filename needs an `LC###` token; the configuration comes from the folder.
2. Add a row to `load_cases.csv` with the loadcase name and its nx, ny, nz.
3. Re-run.

The loadcase name in the sheet must match the `LC###` token. `101` and `LC101` both work. To run a subset while iterating, set `ONLY_LOADCASES = ["LC101"]`.

## Adding or changing sections

Edit `sections.csv`. Each row needs a component, configuration, label, unique section ID within its configuration, `sta_dir`, `lat_dir`, start and end
station, and a centroid. Sections of one component must not overlap and should leave no gaps - both are reported.

## Where to interrogate

| question | look at |
|---|---|
| Which nodes are in a section? | `section_nodes(bodies, loadcase, section_id)` |
| Where did every card row go? | `force_audit` |
| Are the sections covering everything? | `assignment_report` |
| Do the two partitions agree? | `check_partitions_agree` |
| Which case drives sizing? | `station_drivers` |
| Section loads at the centroids | `outputs/section_loads.csv` |
| Point mass forces per case | `outputs/point_mass_loads.csv` |
| Cumulative station values | `outputs/station_diagram.csv` |

## Conventions worth knowing

- All six components are global. 
- Moments in `section_loads.csv` are about each section's own centroid.
