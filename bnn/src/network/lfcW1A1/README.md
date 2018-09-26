# LFCW1A1-PYNQ Network description

HLS description of the LFC neural network using 1 bit weights and 1 activation.

## Utilization reports

Utilization reports using Vivado Design Suite 2018.2 for PYNQ-Z1 and PYNQ-Z2::
```
------------------------------------------------------------------------------------------------
| Design Timing Summary
| ---------------------
------------------------------------------------------------------------------------------------

WNS(ns)      TNS(ns)  TNS Failing Endpoints  TNS Total Endpoints      WHS(ns)      THS(ns) 
-------      -------  ---------------------  -------------------      -------      ------- 
  0.265        0.000                      0                97631        0.012        0.000 


All user specified timing constraints are met.

+----------------------------+-------+-------+-----------+-------+
|          Site Type         |  Used | Fixed | Available | Util% |
+----------------------------+-------+-------+-----------+-------+
| Slice LUTs                 | 28360 |     0 |     53200 | 53.31 |
|   LUT as Logic             | 25359 |     0 |     53200 | 47.67 |
|   LUT as Memory            |  3001 |     0 |     17400 | 17.25 |
|     LUT as Distributed RAM |  2314 |     0 |           |       |
|     LUT as Shift Register  |   687 |     0 |           |       |
| Slice Registers            | 33470 |     0 |    106400 | 31.46 |
|   Register as Flip Flop    | 33470 |     0 |    106400 | 31.46 |
|   Register as Latch        |     0 |     0 |    106400 |  0.00 |
| F7 Muxes                   |   393 |     0 |     26600 |  1.48 |
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
