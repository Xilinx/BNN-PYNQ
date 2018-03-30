# CNV-PYNQ Network description

Network description of the CNV neural network.

## Utilization reports

Utilization reports using Vivado Design Suite 2017.4:
```
------------------------------------------------------------------------------------------------
| Design Timing Summary
| ---------------------
------------------------------------------------------------------------------------------------

WNS(ns)      TNS(ns)  TNS Failing Endpoints  TNS Total Endpoints      WHS(ns)      THS(ns)  THS Failing Endpoints 
-------      -------  ---------------------  -------------------      -------      -------  ---------------------  
  0.281        0.000                      0               112750        0.016        0.000                      0  


All user specified timing constraints are met.

+----------------------------+-------+-------+-----------+-------+
|          Site Type         |  Used | Fixed | Available | Util% |
+----------------------------+-------+-------+-----------+-------+
| Slice LUTs                 | 25770 |     0 |     53200 | 48.44 |
|   LUT as Logic             | 23447 |     0 |     53200 | 44.07 |
|   LUT as Memory            |  2323 |     0 |     17400 | 13.35 |
|     LUT as Distributed RAM |  1938 |     0 |           |       |
|     LUT as Shift Register  |   385 |     0 |           |       |
| Slice Registers            | 42269 |     0 |    106400 | 39.73 |
|   Register as Flip Flop    | 42269 |     0 |    106400 | 39.73 |
|   Register as Latch        |     0 |     0 |    106400 |  0.00 |
| F7 Muxes                   |   908 |     0 |     26600 |  3.41 |
| F8 Muxes                   |   241 |     0 |     13300 |  1.81 |
+----------------------------+-------+-------+-----------+-------+
+-------------------+------+-------+-----------+-------+
|     Site Type     | Used | Fixed | Available | Util% |
+-------------------+------+-------+-----------+-------+
| Block RAM Tile    |  121 |     0 |       140 | 86.43 |
|   RAMB36/FIFO*    |   76 |     0 |       140 | 54.29 |
|     RAMB36E1 only |   76 |       |           |       |
|   RAMB18          |   90 |     0 |       280 | 32.14 |
|     RAMB18E1 only |   90 |       |           |       |
+-------------------+------+-------+-----------+-------+
+----------------+------+-------+-----------+-------+
|    Site Type   | Used | Fixed | Available | Util% |
+----------------+------+-------+-----------+-------+
| DSPs           |   24 |     0 |       220 | 10.91 |
|   DSP48E1 only |   24 |       |           |       |
+----------------+------+-------+-----------+-------+
```

