# CNV-PYNQ Network description

Network description of the CNV neural network.

## Utilization reports

Utilization reports using Vivado Design Suite 2016.1:
```
+----------------------------+-------+-------+-----------+-------+
|          Site Type         |  Used | Fixed | Available | Util% |
+----------------------------+-------+-------+-----------+-------+
| Slice LUTs                 | 29988 |     0 |     53200 | 56.37 |
|   LUT as Logic             | 24289 |     0 |     53200 | 45.66 |
|   LUT as Memory            |  5699 |     0 |     17400 | 32.75 |
|     LUT as Distributed RAM |  2046 |     0 |           |       |
|     LUT as Shift Register  |  3653 |     0 |           |       |
| Slice Registers            | 34489 |     0 |    106400 | 32.41 |
|   Register as Flip Flop    | 34489 |     0 |    106400 | 32.41 |
|   Register as Latch        |     0 |     0 |    106400 |  0.00 |
| F7 Muxes                   |  1714 |     0 |     26600 |  6.44 |
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

