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
  0.119        0.000                      0               105400        0.024        0.000                      0  


All user specified timing constraints are met.

+----------------------------+-------+-------+-----------+-------+
|          Site Type         |  Used | Fixed | Available | Util% |
+----------------------------+-------+-------+-----------+-------+
| Slice LUTs                 | 30516 |     0 |     53200 | 57.36 |
|   LUT as Logic             | 24817 |     0 |     53200 | 46.65 |
|   LUT as Memory            |  5699 |     0 |     17400 | 32.75 |
|     LUT as Distributed RAM |  2046 |     0 |           |       |
|     LUT as Shift Register  |  3653 |     0 |           |       |
| Slice Registers            | 34875 |     0 |    106400 | 32.78 |
|   Register as Flip Flop    | 34875 |     0 |    106400 | 32.78 |
|   Register as Latch        |     0 |     0 |    106400 |  0.00 |
| F7 Muxes                   |  1697 |     0 |     26600 |  6.38 |
| F8 Muxes                   |   818 |     0 |     13300 |  6.15 |
+----------------------------+-------+-------+-----------+-------+
+-------------------+------+-------+-----------+-------+
|     Site Type     | Used | Fixed | Available | Util% |
+-------------------+------+-------+-----------+-------+
| Block RAM Tile    |  116 |     0 |       140 | 82.86 |
|   RAMB36/FIFO*    |   71 |     0 |       140 | 50.71 |
|     RAMB36E1 only |   71 |       |           |       |
|   RAMB18          |   90 |     0 |       280 | 32.14 |
|     RAMB18E1 only |   90 |       |           |       |
+-------------------+------+-------+-----------+-------+
+----------------+------+-------+-----------+-------+
|    Site Type   | Used | Fixed | Available | Util% |
+----------------+------+-------+-----------+-------+
| DSPs           |   32 |     0 |       220 | 14.55 |
|   DSP48E1 only |   32 |       |           |       |
+----------------+------+-------+-----------+-------+
```

