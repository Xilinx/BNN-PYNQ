# LFC-PYNQ Network description

Network description of the LFC neural network.

## Utilization reports

Utilization reports using Vivado Design Suite 2016.1:
```
------------------------------------------------------------------------------------------------
| Design Timing Summary
| ---------------------
------------------------------------------------------------------------------------------------

WNS(ns)      TNS(ns)  TNS Failing Endpoints  TNS Total Endpoints      WHS(ns)      THS(ns)  THS Failing Endpoints
-------      -------  ---------------------  -------------------      -------      -------  ---------------------
  0.217        0.000                      0                96980        0.014        0.000                      0


All user specified timing constraints are met.

+----------------------------+-------+-------+-----------+-------+
|          Site Type         |  Used | Fixed | Available | Util% |
+----------------------------+-------+-------+-----------+-------+
| Slice LUTs                 | 30249 |     0 |     53200 | 56.86 |
|   LUT as Logic             | 27259 |     0 |     53200 | 51.24 |
|   LUT as Memory            |  2990 |     0 |     17400 | 17.18 |
|     LUT as Distributed RAM |  2378 |     0 |           |       |
|     LUT as Shift Register  |   612 |     0 |           |       |
| Slice Registers            | 31760 |     0 |    106400 | 29.85 |
|   Register as Flip Flop    | 31760 |     0 |    106400 | 29.85 |
|   Register as Latch        |     0 |     0 |    106400 |  0.00 |
| F7 Muxes                   |     0 |     0 |     26600 |  0.00 |
| F8 Muxes                   |     0 |     0 |     13300 |  0.00 |
+----------------------------+-------+-------+-----------+-------+
+-------------------+------+-------+-----------+-------+
|     Site Type     | Used | Fixed | Available | Util% |
+-------------------+------+-------+-----------+-------+
| Block RAM Tile    |  112 |     0 |       140 | 80.00 |
|   RAMB36/FIFO*    |    6 |     0 |       140 |  4.29 |
|     RAMB36E1 only |    6 |       |           |       |
|   RAMB18          |  212 |     0 |       280 | 75.71 |
|     RAMB18E1 only |  212 |       |           |       |
+-------------------+------+-------+-----------+-------+
+----------------+------+-------+-----------+-------+
|    Site Type   | Used | Fixed | Available | Util% |
+----------------+------+-------+-----------+-------+
| DSPs           |    5 |     0 |       220 |  2.27 |
|   DSP48E1 only |    5 |       |           |       |
+----------------+------+-------+-----------+-------+
```
