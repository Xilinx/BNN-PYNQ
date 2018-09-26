# LFCW1A2-PYNQ Network description

HLS description of the LFC neural network that uses 1 bit weights and 2 activation.

## Utilization reports

Utilization reports using Vivado Design Suite 2018.2 for PYNQ-Z1 and PYNQ-Z2:
```
------------------------------------------------------------------------------------------------
| Design Timing Summary
| ---------------------
------------------------------------------------------------------------------------------------

WNS(ns)      TNS(ns)  TNS Failing Endpoints  TNS Total Endpoints      WHS(ns)      THS(ns) 
-------      -------  ---------------------  -------------------      -------      -------  
  0.273        0.000                      0               131628        0.020        0.000  


All user specified timing constraints are met.

+----------------------------+-------+-------+-----------+-------+
|          Site Type         |  Used | Fixed | Available | Util% |
+----------------------------+-------+-------+-----------+-------+
| Slice LUTs                 | 48981 |     0 |     53200 | 92.07 |
|   LUT as Logic             | 44190 |     0 |     53200 | 83.06 |
|   LUT as Memory            |  4791 |     0 |     17400 | 27.53 |
|     LUT as Distributed RAM |  4362 |     0 |           |       |
|     LUT as Shift Register  |   429 |     0 |           |       |
| Slice Registers            | 45338 |     0 |    106400 | 42.61 |
|   Register as Flip Flop    | 45338 |     0 |    106400 | 42.61 |
|   Register as Latch        |     0 |     0 |    106400 |  0.00 |
| F7 Muxes                   |   903 |     0 |     26600 |  3.39 |
| F8 Muxes                   |   128 |     0 |     13300 |  0.96 |
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
