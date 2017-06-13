# CNV-PYNQ Network description

Network description of the CNV neural network.

## Utilization reports

Utilization reports using Vivado Design Suite 2016.1:
```
------------------------------------------------------------------------------------------------
| Design Timing Summary
| ---------------------
------------------------------------------------------------------------------------------------

WNS(ns)      TNS(ns)  TNS Failing Endpoints  TNS Total Endpoints      WHS(ns)      THS(ns)  THS Failing Endpoints 
-------      -------  ---------------------  -------------------      -------      -------  ---------------------  
  0.125        0.000                      0               107791        0.026        0.000                      0  


All user specified timing constraints are met.

+----------------------------+-------+-------+-----------+-------+
|          Site Type         |  Used | Fixed | Available | Util% |
+----------------------------+-------+-------+-----------+-------+
| Slice LUTs                 | 30605 |     0 |     53200 | 57.53 |
|   LUT as Logic             | 24666 |     0 |     53200 | 46.36 |
|   LUT as Memory            |  5939 |     0 |     17400 | 34.13 |
|     LUT as Distributed RAM |  2286 |     0 |           |       |
|     LUT as Shift Register  |  3653 |     0 |           |       |
| Slice Registers            | 34645 |     0 |    106400 | 32.56 |
|   Register as Flip Flop    | 34645 |     0 |    106400 | 32.56 |
|   Register as Latch        |     0 |     0 |    106400 |  0.00 |
| F7 Muxes                   |  1697 |     0 |     26600 |  6.38 |
| F8 Muxes                   |   818 |     0 |     13300 |  6.15 |
+----------------------------+-------+-------+-----------+-------+
+-------------------+------+-------+-----------+--------+
|     Site Type     | Used | Fixed | Available |  Util% |
+-------------------+------+-------+-----------+--------+
| Block RAM Tile    |  140 |     0 |       140 | 100.00 |
|   RAMB36/FIFO*    |  104 |     0 |       140 |  74.29 |
|     RAMB36E1 only |  104 |       |           |        |
|   RAMB18          |   72 |     0 |       280 |  25.71 |
|     RAMB18E1 only |   72 |       |           |        |
+-------------------+------+-------+-----------+--------+
+----------------+------+-------+-----------+-------+
|    Site Type   | Used | Fixed | Available | Util% |
+----------------+------+-------+-----------+-------+
| DSPs           |   26 |     0 |       220 | 11.82 |
|   DSP48E1 only |   26 |       |           |       |
+----------------+------+-------+-----------+-------+
```

