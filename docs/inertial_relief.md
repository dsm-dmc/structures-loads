# Applying Secondary Structure Mass: Free Flight

## Inputs
|                                        |                          |
| -------------------------------------- | ------------------------ |
| Structural mass in FEM (shells/solids) | 300 lb                   |
| Secondary structure, as point masses   | 80 lb                    |
| Load factor                            | 5 g                      |
| Customer grid loads                    | aero + inertia, balanced |


Assume no CONM2 for the secondary structure. Forces plus inertia relief only.

## Applied Loads

At each point mass location:

$$F_i = -m_i  LF  g$$

Total: 80 x 5 = 400 lbf. lb-mass x g = lb-force.

## Inertia Relief Response

Assuming the customer's cards balance forces for free flight, so the 400 lbf is the entire residual. Inertia relief derives an acceleration from the mass it sees; 300 lb of structure, since the point masses have no CONM2:

$$a_{ir} = 400 / 300 = 1.33 \text{ g}$$

It returns 400 lbf distributed in proportion to structural mass.

## Resulting Load State


| Location          | Applied  | Returned         | Net          |
| ----------------- | -------- | ---------------- | ------------ |
| Point masses      | -400 lbf | 0                | **-400 lbf** |
| Primary structure | 0        | +400 lbf smeared | **+400 lbf** |


The point masses are outside the mass matrix, so they retain their full load. With CONM2 present, 21% would cancel at those locations.

## Approximation

The 400 lbf returned to primary structure substitutes for the additional load required to hold 5 g at the higher weight.

$$\R = m_p / m_s = 80 / 300 = 27$$

This is the effective load factor shift: 5 g becomes 6.33 g for primary structure. Independent of load factor and direction.

## Dispositioning R

Options:

1. Customer re-trims with secondary structure in the mass case
2. Re-balance airframe inertial loadss: apply 400 lbf as distributed aero/inertial loading on airframe, driving the residual to zero
3. Do nothing, accept conservatism

## Resolving Bending

$$M_R = -LF  g \sum m_i (x_i - x_{cg})$$

Inertia relief absorbs this as angular acceleration, producing load varying linearly along the vehicle. This does not cancel along the length and typically drives fuselage bending harder than the force term.

- Masses balanced about CG → term vanishes regardless of R
- Masses clustered forward or aft → term set by offset, not R

Report $\sum m_i (x_i - x_{cg})$ alongside R for reference.

## Method Summary

1. Forces only at point mass locations, $-m_i LF g$. No CONM2, no GRAV.
2. Inertia relief closes the residual; point masses retain full load.
3. Primary structure receives equal and opposite smear — 27% effective load factor shift.
4. Balance airframe inertial loading with offset loading or get new grid point loads from customer.s
5. Report $\sum m_i (x_i - x_{cg})$; quantify angular term if non-zero.