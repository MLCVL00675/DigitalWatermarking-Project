"""
Remaining Ablations Pipeline Runner (D, E, F, and Final Verification)
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

from run_ablations_suite import (
    run_part_d_block,
    run_part_e_payload,
    run_part_f_heatmap,
)
from run_missing_experiments import verify_and_report_files

def main():
    print("==================================================")
    print(" RUNNING REMAINING ABLATIONS (PARTS D, E, F)")
    print("==================================================")
    
    # Part D
    print("\n[*] Starting Part D: Block Size Ablation...")
    run_part_d_block(workers=14)
    
    # Part E
    print("\n[*] Starting Part E: Payload Scaling Ablation...")
    run_part_e_payload(workers=14)
    
    # Part F
    print("\n[*] Starting Part F: Unified Heatmap Matrix...")
    run_part_f_heatmap()
    
    # Final Verification
    print("\n[*] Running factor verification on all ablation files...")
    verify_and_report_files(results_dir="results_v3")
    print("\n[OK] All ablation tasks finished successfully.")

if __name__ == "__main__":
    main()

