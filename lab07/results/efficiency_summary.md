| Pillar | Metric | baseline | global pruned 90% | structured 50% filters | plain student | distilled student |
|---|---|---:|---:|---:|---:|---:|
| quality | test accuracy | 0.9202 | 0.9275 | 0.9268 | 0.9099 | 0.9061 |
| data | training samples seen, incl. baseline it depends on | 550,000 | 715,000 | 715,000 | 550,000 | 1,100,000 |
| data | needs the trained baseline | no | yes (prunes it) | yes (prunes it) | no | yes (soft labels) |
| model | parameters (stored) | 997,434 | 997,434 | 476,474 | 105,866 | 105,866 |
| model | non-zero parameters | 997,434 | 100,084 | 476,474 | 105,866 | 105,866 |
| model | size, dense float32 (MB) | 3.9928 | 3.9928 | 1.9090 | 0.4261 | 0.4261 |
| model | size, best sparse format (MB) | 4.1186 | 0.5292 | 1.9697 | 0.4402 | 0.4402 |
| model | size, dense + gzip (MB) | 3.7117 | 0.7557 | 1.7766 | 0.3955 | 0.3956 |
| compute | MACs per image, dense kernels | 8,355,744 | 8,355,744 | 2,372,256 | 1,117,056 | 1,117,056 |
| compute | MACs per image, ideal sparse kernel | 8,355,744 | 2,085,289 | 2,372,256 | 1,117,056 | 1,117,056 |
| compute | latency batch 1, 1 thread (ms, median) | 0.3642 | 0.3623 | 0.1991 | 0.1485 | 0.1486 |
| compute | throughput batch 128, 1 thread (samples/s) | 2524.5833 | 2525.4643 | 6476.8657 | 9037.4071 | 9096.3873 |
| compute | training time incl. baseline (s) | 303.0095 | 393.2012 | 364.1114 | 132.9005 | 535.4296 |
| energy | inference energy per image (mJ, estimate at 5 W) | 1.8209 | 1.8117 | 0.9954 | 0.7424 | 0.7428 |
| energy | inference energy relative to baseline (= latency ratio) | 1.0000 | 0.9949 | 0.5467 | 0.4077 | 0.4079 |
| energy | training energy (kJ, estimate at 20 W) | 6.0602 | 7.8640 | 7.2822 | 2.6580 | 10.7086 |
