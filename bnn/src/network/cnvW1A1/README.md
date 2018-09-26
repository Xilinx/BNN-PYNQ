# CNVW1A1-PYNQ Network description

Network description of the CNV neural network using 1 bit weights and 1 activation.

## Utilization reports

Utilization reports using Vivado Design Suite 2018.2 for PYNQ-Z1 and PYNQ-Z2:
```
------------------------------------------------------------------------------------------------
| Design Timing Summary
| ---------------------
------------------------------------------------------------------------------------------------

WNS(ns)      TNS(ns)  TNS Failing Endpoints  TNS Total Endpoints      WHS(ns)      THS(ns) 
-------      -------  ---------------------  -------------------      -------      ------- 
  0.285        0.000                      0               108705        0.020        0.000 


All user specified timing constraints are met.

+----------------------------+-------+-------+-----------+-------+
|          Site Type         |  Used | Fixed | Available | Util% |
+----------------------------+-------+-------+-----------+-------+
| Slice LUTs                 | 26072 |     0 |     53200 | 49.01 |
|   LUT as Logic             | 24047 |     0 |     53200 | 45.20 |
|   LUT as Memory            |  2025 |     0 |     17400 | 11.64 |
|     LUT as Distributed RAM |  1578 |     0 |           |       |
|     LUT as Shift Register  |   447 |     0 |           |       |
| Slice Registers            | 41312 |     0 |    106400 | 38.83 |
|   Register as Flip Flop    | 41312 |     0 |    106400 | 38.83 |
|   Register as Latch        |     0 |     0 |    106400 |  0.00 |
| F7 Muxes                   |   892 |     0 |     26600 |  3.35 |
| F8 Muxes                   |   241 |     0 |     13300 |  1.81 |
+----------------------------+-------+-------+-----------+-------+
+-------------------+------+-------+-----------+-------+
|     Site Type     | Used | Fixed | Available | Util% |
+-------------------+------+-------+-----------+-------+
| Block RAM Tile    |  124 |     0 |       140 | 88.57 |
|   RAMB36/FIFO*    |   76 |     0 |       140 | 54.29 |
|     RAMB36E1 only |   76 |       |           |       |
|   RAMB18          |   96 |     0 |       280 | 34.29 |
|     RAMB18E1 only |   96 |       |           |       |
+-------------------+------+-------+-----------+-------+
+----------------+------+-------+-----------+-------+
|    Site Type   | Used | Fixed | Available | Util% |
+----------------+------+-------+-----------+-------+
| DSPs           |   24 |     0 |       220 | 10.91 |
|   DSP48E1 only |   24 |       |           |       |
+----------------+------+-------+-----------+-------+
```
