# LFC-PYNQ Network description

Network description of the LFC neural network.

## Utilization reports

Utilization reports using Vivado Design Suite 2017.4:
```
------------------------------------------------------------------------------------------------
| Design Timing Summary
| ---------------------
------------------------------------------------------------------------------------------------

WNS(ns)      TNS(ns)  TNS Failing Endpoints  TNS Total Endpoints      WHS(ns)      THS(ns)  THS Failing Endpoints
-------      -------  ---------------------  -------------------      -------      -------  ---------------------
  0.221        0.000                      0                93621        0.018        0.000                      0


All user specified timing constraints are met.

+----------------------------+-------+-------+-----------+-------+
|          Site Type         |  Used | Fixed | Available | Util% |
+----------------------------+-------+-------+-----------+-------+
| Slice LUTs                 | 25358 |     0 |     53200 | 47.67 |
|   LUT as Logic             | 22376 |     0 |     53200 | 42.06 |
|   LUT as Memory            |  2982 |     0 |     17400 | 17.14 |
|     LUT as Distributed RAM |  2314 |     0 |           |       |
|     LUT as Shift Register  |   668 |     0 |           |       |
| Slice Registers            | 31500 |     0 |    106400 | 29.61 |
|   Register as Flip Flop    | 31500 |     0 |    106400 | 29.61 |
|   Register as Latch        |     0 |     0 |    106400 |  0.00 |
| F7 Muxes                   |   394 |     0 |     26600 |  1.48 |
| F8 Muxes                   |    64 |     0 |     13300 |  0.48 |
+----------------------------+-------+-------+-----------+-------+
+-------------------+------+-------+-----------+-------+
|     Site Type     | Used | Fixed | Available | Util% |
+-------------------+------+-------+-----------+-------+
| Block RAM Tile    |  110 |     0 |       140 | 78.57 |
|   RAMB36/FIFO*    |    6 |     0 |       140 |  4.29 |
|     RAMB36E1 only |    6 |       |           |       |
|   RAMB18          |  208 |     0 |       280 | 74.29 |
|     RAMB18E1 only |  208 |       |           |       |
+-------------------+------+-------+-----------+-------+
+----------------+------+-------+-----------+-------+
|    Site Type   | Used | Fixed | Available | Util% |
+----------------+------+-------+-----------+-------+
| DSPs           |    4 |     0 |       220 |  1.82 |
|   DSP48E1 only |    4 |       |           |       |
+----------------+------+-------+-----------+-------+
```
