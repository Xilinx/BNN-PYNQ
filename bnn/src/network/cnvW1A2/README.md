# CNVW1A2-PYNQ Network description

Network description of the CNV neural network using 1 bit weights and 2 acitvation.

## Utilization reports

Utilization reports using Vivado Design Suite 2018.2 for PYNQ-Z1 and PYNQ-Z2:
```
------------------------------------------------------------------------------------------------
| Design Timing Summary
| ---------------------
------------------------------------------------------------------------------------------------

WNS(ns)      TNS(ns)  TNS Failing Endpoints  TNS Total Endpoints      WHS(ns)      THS(ns) 
-------      -------  ---------------------  -------------------      -------      ------- 
  0.307        0.000                      0               146986        0.011        0.000 


All user specified timing constraints are met.

+----------------------------+-------+-------+-----------+-------+
|          Site Type         |  Used | Fixed | Available | Util% |
+----------------------------+-------+-------+-----------+-------+
| Slice LUTs                 | 40321 |     0 |     53200 | 75.79 |
|   LUT as Logic             | 36545 |     0 |     53200 | 68.69 |
|   LUT as Memory            |  3776 |     0 |     17400 | 21.70 |
|     LUT as Distributed RAM |  3370 |     0 |           |       |
|     LUT as Shift Register  |   406 |     0 |           |       |
| Slice Registers            | 56276 |     0 |    106400 | 52.89 |
|   Register as Flip Flop    | 56276 |     0 |    106400 | 52.89 |
|   Register as Latch        |     0 |     0 |    106400 |  0.00 |
| F7 Muxes                   |  1757 |     0 |     26600 |  6.61 |
| F8 Muxes                   |   482 |     0 |     13300 |  3.62 |
+----------------------------+-------+-------+-----------+-------+
+-------------------+-------+-------+-----------+-------+
|     Site Type     |  Used | Fixed | Available | Util% |
+-------------------+-------+-------+-----------+-------+
| Block RAM Tile    | 131.5 |     0 |       140 | 93.93 |
|   RAMB36/FIFO*    |    90 |     0 |       140 | 64.29 |
|     RAMB36E1 only |    90 |       |           |       |
|   RAMB18          |    83 |     0 |       280 | 29.64 |
|     RAMB18E1 only |    83 |       |           |       |
+-------------------+-------+-------+-----------+-------+
+----------------+------+-------+-----------+-------+
|    Site Type   | Used | Fixed | Available | Util% |
+----------------+------+-------+-----------+-------+
| DSPs           |   26 |     0 |       220 | 11.82 |
|   DSP48E1 only |   26 |       |           |       |
+----------------+------+-------+-----------+-------+
```
