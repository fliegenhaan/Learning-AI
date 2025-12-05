"""
Generate Submission File for Kaggle
Script sederhana untuk generate submission menggunakan ModelStorage
"""

import sys
import glob
from pathlib import Path

# Import modules
from data import DataLoader
from preprocessing import ModelPipeline
from storage import ModelStorage


def main():
    """Main function untuk generate submission"""

    print("\n" + "=" * 40)
    print("KAGGLE SUBMISSION GENERATOR")
    print("=" * 40)

    # LOAD DATA
    print("\n[STEP 1] Loading data...")

    train_url = "https://drive.google.com/uc?id=1wzTvPSwjAK5PN0iCWEXy92_tim-5ggjs"
    test_url = "https://drive.google.com/uc?id=1ZoKNPeUAIIFIZHoKaY6_4R_fUqDue0HM"

    loader = DataLoader(train_url=train_url, test_url=test_url)
    train_df, test_df = loader.load_data()

    print(f"  Training data shape: {train_df.shape}")
    print(f"  Test data shape: {test_df.shape}")

    # PREPARE PIPELINE
    print("\n[STEP 2] Preparing preprocessing pipeline...")

    # Split train data untuk fit pipeline
    X_train, y_train = loader.split_features_target(train_df, target_col='Target')

    # Build dan fit pipeline
    pipeline_builder = ModelPipeline()
    pipeline = pipeline_builder.build_pipeline(
        outlier_factor=1.5,
        pca_components=0.95,
        use_normalizer=True
    )

    print("  Fitting pipeline on training data...")
    pipeline.fit(X_train)
    print("  Pipeline ready!")

    # LOAD TRAINED MODEL
    print("\n[STEP 3] Loading trained model...")

    # List available models from multiple locations
    print("  Searching for model files...")

    # Search in current directory and storage directory
    search_paths = [
        "*.pkl",                    # Current directory
        "storage/*.pkl",            # Storage folder
        "src/storage/*.pkl",        # src/storage folder
    ]

    pkl_files = []
    for pattern in search_paths:
        files = glob.glob(pattern)
        if files:
            pkl_files.extend(files)

    # Remove duplicates
    pkl_files = list(set(pkl_files))

    if not pkl_files:
        print("\n  No .pkl files found!")
        print("  Please train a model first using: python src/main.py")
        print("\n  Searched locations:")
        print("    - Current directory")
        print("    - storage/")
        print("    - src/storage/")
        return

    print("\n  Available model files:")
    for i, f in enumerate(pkl_files, 1):
        print(f"  {i}. {f}")

    # Ask user to choose
    choice = input("\nPilih model file (1/2/3/...) atau ketik nama file: ").strip()

    if choice.isdigit() and 1 <= int(choice) <= len(pkl_files):
        model_path = pkl_files[int(choice) - 1]
    else:
        # User typed filename, try to find it
        if not choice.endswith('.pkl'):
            choice = f"{choice}.pkl"

        # Search in all locations
        found = False
        for pattern in search_paths:
            search_dir = pattern.rsplit('/', 1)[0] if '/' in pattern else '.'
            potential_path = f"{search_dir}/{choice}" if search_dir != '.' else choice
            if Path(potential_path).exists():
                model_path = potential_path
                found = True
                break

        if not found:
            model_path = choice

    try:
        model = ModelStorage.load_model_for_prediction(model_path)
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        return

    # GENERATE SUBMISSION
    print("\n[STEP 4] Generating submission...")
    submission_filename = input("  Nama file submission (default: submission.csv): ").strip() or "submission.csv"

    if not submission_filename.endswith('.csv'):
        submission_filename += '.csv'

    # Generate submission using ModelStorage
    submission_df = ModelStorage.generate_submission(
        model=model,
        test_data=test_df,
        pipeline=pipeline,
        filename=submission_filename,
        id_column='id'
    )

    # Show sample
    print(f"\n  Sample submission (first 10 rows):")
    print(submission_df.head(10))

    print("\nSubmission generation completed!")
    print(f"  File ready for Kaggle: {submission_filename}")
    print(f"\n  Next steps:")
    print(f"    1. Go to Kaggle competition page")
    print(f"    2. Click 'Submit Predictions'")
    print(f"    3. Upload {submission_filename}")
    print(f"    4. Check your leaderboard score!")


if __name__ == "__main__":
    main()
