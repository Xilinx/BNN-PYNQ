# CNVW2A2-PYNQ Network description

Network description of the CNV neural network using 2 bit weights and 2 activation.

## Utilization reports

Utilization reports using Vivado Design Suite 2018.2 for PYNQ-Z1 and PYNQ-Z2:
```
------------------------------------------------------------------------------------------------
| Design Timing Summary
| ---------------------
------------------------------------------------------------------------------------------------

WNS(ns)      TNS(ns)  TNS Failing Endpoints  TNS Total Endpoints      WHS(ns)      THS(ns)
-------      -------  ---------------------  -------------------      -------      ------- 
  0.172        0.000                      0               172305        0.017        0.000 


All user specified timing constraints are met.

+----------------------------+-------+-------+-----------+-------+
|          Site Type         |  Used | Fixed | Available | Util% |
+----------------------------+-------+-------+-----------+-------+
| Slice LUTs                 | 37302 |     0 |     53200 | 70.12 |
|   LUT as Logic             | 29865 |     0 |     53200 | 56.14 |
|   LUT as Memory            |  7437 |     0 |     17400 | 42.74 |
|     LUT as Distributed RAM |  6970 |     0 |           |       |
|     LUT as Shift Register  |   467 |     0 |           |       |
| Slice Registers            | 51386 |     0 |    106400 | 48.30 |
|   Register as Flip Flop    | 51386 |     0 |    106400 | 48.30 |
|   Register as Latch        |     0 |     0 |    106400 |  0.00 |
| F7 Muxes                   |  3072 |     0 |     26600 | 11.55 |
| F8 Muxes                   |   956 |     0 |     13300 |  7.19 |
+----------------------------+-------+-------+-----------+-------+
+-------------------+------+-------+-----------+--------+
|     Site Type     | Used | Fixed | Available |  Util% |
+-------------------+------+-------+-----------+--------+
| Block RAM Tile    |  140 |     0 |       140 | 100.00 |
|   RAMB36/FIFO*    |  138 |     0 |       140 |  98.57 |
|     RAMB36E1 only |  138 |       |           |        |
|   RAMB18          |    4 |     0 |       280 |   1.43 |
|     RAMB18E1 only |    4 |       |           |        |
+-------------------+------+-------+-----------+--------+
+----------------+------+-------+-----------+-------+
|    Site Type   | Used | Fixed | Available | Util% |
+----------------+------+-------+-----------+-------+
| DSPs           |   32 |     0 |       220 | 14.55 |
|   DSP48E1 only |   32 |       |           |       |
+----------------+------+-------+-----------+-------+
```
