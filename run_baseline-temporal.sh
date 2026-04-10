#!/bin/bash
cd /root/confereai
python3 scripts/baseline_isolation_forest.py --contamination 0.05 --methods if,ocsvm --output data/baseline_results_temporal
