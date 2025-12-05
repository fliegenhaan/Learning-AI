import sys
import os

# Tambahkan path src ke sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

print("\n" + "=" * 50)
print("MODEL COMPARISON: FROM SCRATCH vs SKLEARN")
print("=" * 50)
print("\nPerbandingan model:")
print("  1. Logistic Regression")
print("  2. SVM")
print("  3. Decision Tree")

# Tanya user mau run yang mana
print("\nPilih test yang ingin dijalankan:")
print("1. Logistic Regression")
print("2. SVM")
print("3. Decision Tree")
print("4. Semua (LogReg + SVM + DTL)")
print("5. Exit")

choice = input("\nPilihan Anda (1/2/3/4/5): ").strip()

if choice == "1":
    print("\n[RUNNING] Logistic Regression Comparison")
    from test_logres import main as logres_main
    logres_main()

elif choice == "2":
    print("\n[RUNNING] SVM Comparison")
    from test_svm import main as svm_main
    svm_main()

elif choice == "3":
    print("\n[RUNNING] Decision Tree Comparison")
    from test_dtl import main as dtl_main
    dtl_main()

elif choice == "4":
    print("\n[RUNNING] Full Comparison (LogReg + SVM + DTL)\n")

    print("[PART 1/3] Logistic Regression")
    from test_logres import main as logres_main
    logres_main()

    print("\n[PART 2/3] SVM")
    from test_svm import main as svm_main
    svm_main()

    print("\n[PART 3/3] Decision Tree")
    from test_dtl import main as dtl_main
    dtl_main()

    print("\nAll comparisons completed!")
    print("\n  Comparison plots saved:")
    print("    - logistic_regression_comparison_final.png")
    print("    - svm_comparison_rbf.png")
    print("    - svm_comparison_linear.png")
    print("    - dtl_comparison_final.png")

elif choice == "5":
    print("\nExiting...")
    sys.exit(0)

else:
    print("\nPilihan tidak valid. Exiting...")
    sys.exit(1)
