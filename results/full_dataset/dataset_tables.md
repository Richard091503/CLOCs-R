## Scenes / frames by split and condition

| Split | highway | urban | night | rain | All |
|---|---|---|---|---|---|
| train | 63 / 8,924 | 46 / 6,525 | 22 / 3,117 | 3 / 428 | 134 / 18,994 |
| holdout | 7 / 990 | 6 / 851 | 3 / 425 | 1 / 142 | 17 / 2,408 |
| val | 12 / 2,252 | 9 / 2,040 | 2 / 444 | 2 / 445 | 25 / 5,181 |
| all | 82 / 12,166 | 61 / 9,416 | 27 / 3,986 | 6 / 1,015 | 176 / 26,583 |

## Scored cars by split, condition and distance

Scored cars: CAR, VAN and PICKUP labels inside the evaluation region (0-204.8 m ahead, within 25.6 m sideways) whose center projects into the front camera. Beyond 50 m means 50-200 m, as in the evaluation.

| Split | Condition | Cars | 0-30 m | 30-50 m | 50-100 m | 100-200 m | Beyond 50 m | Share beyond 50 m | Far cars per frame | Frames with a far car |
|---|---|---|---|---|---|---|---|---|---|---|
| train | highway | 23,812 | 4,897 | 5,092 | 10,387 | 3,436 | 13,823 | 58.1% | 1.55 | 80.8% |
| train | urban | 57,223 | 15,151 | 16,321 | 19,801 | 5,886 | 25,687 | 44.9% | 3.94 | 94.1% |
| train | night | 13,579 | 2,085 | 2,833 | 6,409 | 2,234 | 8,643 | 63.6% | 2.77 | 92.4% |
| train | rain | 1,010 | 45 | 193 | 549 | 223 | 772 | 76.4% | 1.80 | 100.0% |
| train | all | 95,624 | 22,178 | 24,439 | 37,146 | 11,779 | 48,925 | 51.2% | 2.58 | 87.7% |
| holdout | highway | 2,984 | 405 | 632 | 1,482 | 461 | 1,943 | 65.1% | 1.96 | 92.3% |
| holdout | urban | 9,481 | 2,622 | 2,744 | 3,651 | 464 | 4,115 | 43.4% | 4.84 | 90.0% |
| holdout | night | 1,912 | 664 | 521 | 714 | 13 | 727 | 38.0% | 1.71 | 77.4% |
| holdout | rain | 142 | 0 | 142 | 0 | 0 | 0 | 0.0% | 0.00 | 0.0% |
| holdout | all | 14,519 | 3,691 | 4,039 | 5,847 | 938 | 6,785 | 46.7% | 2.82 | 83.4% |
| val | highway | 3,044 | 361 | 310 | 653 | 1,680 | 2,333 | 76.6% | 1.04 | 77.4% |
| val | urban | 14,625 | 2,113 | 1,160 | 6,255 | 5,097 | 11,352 | 77.6% | 5.56 | 98.5% |
| val | night | 1,943 | 382 | 212 | 630 | 719 | 1,349 | 69.4% | 3.04 | 100.0% |
| val | rain | 1,806 | 255 | 213 | 610 | 728 | 1,338 | 74.1% | 3.01 | 95.7% |
| val | all | 21,418 | 3,111 | 1,895 | 8,148 | 8,224 | 16,372 | 76.4% | 3.16 | 89.2% |
| all | highway | 29,840 | 5,663 | 6,034 | 12,522 | 5,577 | 18,099 | 60.7% | 1.49 | 81.1% |
| all | urban | 81,329 | 19,886 | 20,225 | 29,707 | 11,447 | 41,154 | 50.6% | 4.37 | 94.7% |
| all | night | 17,434 | 3,131 | 3,566 | 7,753 | 2,966 | 10,719 | 61.5% | 2.69 | 91.6% |
| all | rain | 2,958 | 300 | 548 | 1,159 | 951 | 2,110 | 71.3% | 2.08 | 84.1% |
| all | all | 131,561 | 28,980 | 30,373 | 51,141 | 20,941 | 72,082 | 54.8% | 2.71 | 87.6% |

## Labelled objects by class, all splits

| Class | Labels | In evaluation region and camera view | Of those, beyond 50 m | Role in this study |
|---|---|---|---|---|
| CAR | 363,005 | 127,005 | 68,457 | scored car |
| TRUCK | 25,024 | 10,337 | 8,012 | ignored (neither positive nor negative) |
| PEDESTRIAN | 14,060 | 5,511 | 2,376 | not used |
| BUS | 7,740 | 2,937 | 1,336 | ignored (neither positive nor negative) |
| VAN | 6,249 | 3,667 | 2,932 | scored car |
| MOTORCYCLE | 2,611 | 823 | 252 | not used |
| PICKUP | 2,060 | 889 | 693 | scored car |
| RIDER | 1,994 | 775 | 145 | not used |
| BICYCLE | 1,466 | 468 | 285 | not used |
| TRAILER | 725 | 286 | 219 | ignored (neither positive nor negative) |
| OTHER-OBJECT | 558 | 260 | 67 | not used |
| OTHER-RIDEABLE | 515 | 94 | 28 | not used |
| TRAIN | 499 | 49 | 31 | not used |
| SHOPPING-CART | 406 | 75 | 50 | not used |

## Sensor evidence on scored cars: LiDAR

| Condition | Distance | Cars | Any point | 5+ points | Median points |
|---|---|---|---|---|---|
| highway | 0-30 m | 5,663 | 100.0% | 100.0% | 826 |
| highway | 30-50 m | 6,034 | 99.8% | 99.8% | 165 |
| highway | 50-100 m | 12,522 | 99.4% | 97.8% | 38 |
| highway | 100-200 m | 5,577 | 94.5% | 76.3% | 9 |
| highway | **beyond 50 m** | 18,099 | 97.9% | 91.2% | 27 |
| urban | 0-30 m | 19,886 | 99.5% | 99.2% | 662 |
| urban | 30-50 m | 20,225 | 95.7% | 93.5% | 99 |
| urban | 50-100 m | 29,707 | 83.3% | 72.1% | 18 |
| urban | 100-200 m | 11,447 | 60.0% | 36.2% | 2 |
| urban | **beyond 50 m** | 41,154 | 76.8% | 62.1% | 10 |
| night | 0-30 m | 3,131 | 100.0% | 99.9% | 974 |
| night | 30-50 m | 3,566 | 98.1% | 97.4% | 145 |
| night | 50-100 m | 7,753 | 86.0% | 79.4% | 29 |
| night | 100-200 m | 2,966 | 64.0% | 36.9% | 3 |
| night | **beyond 50 m** | 10,719 | 79.9% | 67.6% | 14 |
| rain | 0-30 m | 300 | 100.0% | 100.0% | 905 |
| rain | 30-50 m | 548 | 99.5% | 97.6% | 174 |
| rain | 50-100 m | 1,159 | 92.7% | 89.0% | 29 |
| rain | 100-200 m | 951 | 79.9% | 58.3% | 6 |
| rain | **beyond 50 m** | 2,110 | 86.9% | 75.1% | 15 |
| all | 0-30 m | 28,980 | 99.6% | 99.4% | 726 |
| all | 30-50 m | 30,373 | 96.8% | 95.2% | 125 |
| all | 50-100 m | 51,141 | 87.9% | 79.9% | 26 |
| all | 100-200 m | 20,941 | 70.6% | 48.0% | 4 |
| all | **beyond 50 m** | 72,082 | 82.9% | 70.6% | 15 |

## Sensor evidence on scored cars: Radar (CLOCs-R gate, 3 sweeps)

| Condition | Distance | Cars | Any return | Median returns, if any | Moving (Doppler > 1 m/s), if any |
|---|---|---|---|---|---|
| highway | 0-30 m | 5,663 | 57.9% | 1 | 99.8% |
| highway | 30-50 m | 6,034 | 83.8% | 2 | 98.2% |
| highway | 50-100 m | 12,522 | 86.7% | 2 | 92.0% |
| highway | 100-200 m | 5,577 | 68.3% | 2 | 80.7% |
| highway | **beyond 50 m** | 18,099 | 81.0% | 2 | 89.1% |
| urban | 0-30 m | 19,886 | 51.0% | 3 | 79.1% |
| urban | 30-50 m | 20,225 | 52.3% | 3 | 65.7% |
| urban | 50-100 m | 29,707 | 51.1% | 3 | 64.2% |
| urban | 100-200 m | 11,447 | 50.8% | 3 | 50.9% |
| urban | **beyond 50 m** | 41,154 | 51.0% | 3 | 60.5% |
| night | 0-30 m | 3,131 | 65.4% | 3 | 82.8% |
| night | 30-50 m | 3,566 | 58.1% | 3 | 79.4% |
| night | 50-100 m | 7,753 | 59.9% | 3 | 63.8% |
| night | 100-200 m | 2,966 | 65.0% | 3 | 63.1% |
| night | **beyond 50 m** | 10,719 | 61.3% | 3 | 63.6% |
| rain | 0-30 m | 300 | 38.0% | 2 | 98.2% |
| rain | 30-50 m | 548 | 91.4% | 2 | 98.4% |
| rain | 50-100 m | 1,159 | 89.0% | 2 | 98.5% |
| rain | 100-200 m | 951 | 70.5% | 3 | 87.6% |
| rain | **beyond 50 m** | 2,110 | 80.7% | 2 | 94.2% |
| all | 0-30 m | 28,980 | 53.8% | 2 | 84.1% |
| all | 30-50 m | 30,373 | 59.9% | 3 | 77.2% |
| all | 50-100 m | 51,141 | 62.0% | 3 | 74.8% |
| all | 100-200 m | 20,941 | 58.3% | 3 | 64.1% |
| all | **beyond 50 m** | 72,082 | 61.0% | 3 | 71.8% |

## Sensor evidence on scored cars: Camera (projected 3D box)

| Condition | Distance | Cars | Median height (px) | Under 25 px |
|---|---|---|---|---|
| highway | 0-30 m | 5,663 | 99 | 0.0% |
| highway | 30-50 m | 6,034 | 45 | 0.3% |
| highway | 50-100 m | 12,522 | 23 | 59.4% |
| highway | 100-200 m | 5,577 | 14 | 97.8% |
| highway | **beyond 50 m** | 18,099 | 20 | 71.2% |
| urban | 0-30 m | 19,886 | 97 | 0.1% |
| urban | 30-50 m | 20,225 | 43 | 4.7% |
| urban | 50-100 m | 29,707 | 23 | 56.1% |
| urban | 100-200 m | 11,447 | 13 | 97.8% |
| urban | **beyond 50 m** | 41,154 | 20 | 67.7% |
| night | 0-30 m | 3,131 | 118 | 0.0% |
| night | 30-50 m | 3,566 | 43 | 1.8% |
| night | 50-100 m | 7,753 | 24 | 51.5% |
| night | 100-200 m | 2,966 | 12 | 98.3% |
| night | **beyond 50 m** | 10,719 | 21 | 64.5% |
| rain | 0-30 m | 300 | 114 | 0.0% |
| rain | 30-50 m | 548 | 45 | 0.0% |
| rain | 50-100 m | 1,159 | 24 | 59.4% |
| rain | 100-200 m | 951 | 13 | 100.0% |
| rain | **beyond 50 m** | 2,110 | 15 | 77.7% |
| all | 0-30 m | 28,980 | 99 | 0.1% |
| all | 30-50 m | 30,373 | 44 | 3.4% |
| all | 50-100 m | 51,141 | 23 | 56.3% |
| all | 100-200 m | 20,941 | 13 | 98.0% |
| all | **beyond 50 m** | 72,082 | 20 | 68.4% |

## Sensor evidence on scored cars: Combined

| Condition | Distance | Cars | Radar but no LiDAR | Neither LiDAR nor radar |
|---|---|---|---|---|
| highway | 0-30 m | 5,663 | 0.0% | 0.0% |
| highway | 30-50 m | 6,034 | 0.1% | 0.1% |
| highway | 50-100 m | 12,522 | 0.1% | 0.5% |
| highway | 100-200 m | 5,577 | 2.3% | 3.2% |
| highway | **beyond 50 m** | 18,099 | 0.8% | 1.3% |
| urban | 0-30 m | 19,886 | 0.2% | 0.4% |
| urban | 30-50 m | 20,225 | 0.6% | 3.8% |
| urban | 50-100 m | 29,707 | 4.6% | 12.1% |
| urban | 100-200 m | 11,447 | 14.5% | 25.6% |
| urban | **beyond 50 m** | 41,154 | 7.4% | 15.8% |
| night | 0-30 m | 3,131 | 0.0% | 0.0% |
| night | 30-50 m | 3,566 | 0.5% | 1.4% |
| night | 50-100 m | 7,753 | 6.4% | 7.7% |
| night | 100-200 m | 2,966 | 19.1% | 16.9% |
| night | **beyond 50 m** | 10,719 | 9.9% | 10.2% |
| rain | 0-30 m | 300 | 0.0% | 0.0% |
| rain | 30-50 m | 548 | 0.0% | 0.5% |
| rain | 50-100 m | 1,159 | 4.0% | 3.4% |
| rain | 100-200 m | 951 | 8.9% | 11.1% |
| rain | **beyond 50 m** | 2,110 | 6.2% | 6.9% |
| all | 0-30 m | 28,980 | 0.1% | 0.3% |
| all | 30-50 m | 30,373 | 0.4% | 2.7% |
| all | 50-100 m | 51,141 | 3.8% | 8.4% |
| all | 100-200 m | 20,941 | 11.6% | 17.7% |
| all | **beyond 50 m** | 72,082 | 6.1% | 11.1% |

## Per frame, median

| Condition | Ego speed | LiDAR points | Radar returns (3 sweeps) |
|---|---|---|---|
| highway | 94 km/h | 20,723 | 108 |
| urban | 38 km/h | 21,214 | 74 |
| night | 51 km/h | 22,146 | 90 |
| rain | 112 km/h | 21,045 | 125 |
| all | 64 km/h | 21,113 | 90 |
