def main(
    data_input_path="your_input_file.csv",
    output_directory="Graduate_indicators_preprocessing",
    ordinal_candidates=None,
    low_var_cols=None,
    sep=","
):
    import pandas as pd
    from pathlib import Path

    # --- 1. Load Dataset ---
    try:
        df = pd.read_csv(data_input_path, sep=sep)
        print(
            f"Dataset loaded from '{data_input_path}' with {df.shape[0]} rows and {df.shape[1]} columns."
        )
    except FileNotFoundError:
        print(f"Error: The file '{data_input_path}' was not found.")
        return

    # --- 2. Feature Detection ---
    def auto_detect_feature_types(df, target, ordinal_candidates=None):
        """
        Automatically detect feature types in the dataframe.
        Returns a dictionary with keys: binary, ordinal, continuous, categorical.
        """
        binary = []
        ordinal = []
        continuous = []
        categorical = []
        for col in df.columns:
            if col == target:
                continue
            if ordinal_candidates and col in ordinal_candidates:
                ordinal.append(col)
            elif df[col].nunique() == 2:
                binary.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                continuous.append(col)
            else:
                categorical.append(col)
        return {
            "binary": binary,
            "ordinal": ordinal,
            "continuous": continuous,
            "categorical": categorical,
        }

    feature_types = auto_detect_feature_types(df, target="Status", ordinal_candidates=ordinal_candidates)
    print("Binary features:", feature_types["binary"])
    print("Ordinal features:", feature_types["ordinal"])
    print("Continuous features:", feature_types["continuous"])
    print("Categorical features:", feature_types["categorical"])

    # --- 3. Split Data ---
    X = df.drop("Status", axis=1)
    y = df["Status"]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data split: Training {X_train.shape}, Test {X_test.shape}")
    print(f"Training target distribution:\n{y_train.value_counts(normalize=True).round(3)}")

    # --- 4. Preprocessing Pipeline ---
    def get_preprocessor(feature_types):
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
        from sklearn.compose import ColumnTransformer

        transformers = []

        if feature_types["continuous"]:
            transformers.append(
                ("cont", 
                 make_pipeline(
                    SimpleImputer(strategy="mean"),
                    StandardScaler()
                 ), feature_types["continuous"])
            )
        if feature_types["binary"]:
            transformers.append(
                ("bin", 
                 SimpleImputer(strategy="most_frequent"),
                 feature_types["binary"])
            )
        if feature_types["ordinal"]:
            transformers.append(
                ("ord", 
                 make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OrdinalEncoder()
                 ), feature_types["ordinal"])
            )
        if feature_types["categorical"]:
            transformers.append(
                ("cat",
                 make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore")
                 ), feature_types["categorical"])
            )

        return ColumnTransformer(transformers)

    # Helper for pipeline construction without import issues
    from sklearn.pipeline import make_pipeline

    # 1. Preprocessor (ColumnTransformer, not Pipeline)
    preprocessor = get_preprocessor(feature_types)

    # 2. Main ImbalancedPipeline
    from feature_engine.selection import DropConstantFeatures, DropCorrelatedFeatures
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbalancedPipeline

    pipeline_steps = []
    # Drop low variance/constant columns if specified
    if low_var_cols and len(low_var_cols) > 0:
        pipeline_steps.append(("low_var", DropConstantFeatures(variables=low_var_cols)))
    pipeline_steps += [
        ("preprocess", preprocessor),
        ("feature_selection", DropCorrelatedFeatures(threshold=0.8, method="pearson")),
        ("smote", SMOTE(random_state=42, sampling_strategy="auto")),
    ]
    pipeline = ImbalancedPipeline(pipeline_steps)

    print("Applying preprocessing and SMOTE to training data...")
    X_train_processed, y_train_processed = pipeline.fit_resample(X_train, y_train)
    # For test set: only transform, not resample
    X_test_processed = pipeline[:-1].transform(X_test)
    final_feature_names = pipeline.named_steps["feature_selection"].get_feature_names_out()

    print(f"Preprocessing complete. Training data shape: {X_train_processed.shape}")
    print(f"Test data shape: {X_test_processed.shape}")
    print(
        f"Training target distribution after SMOTE:\n{pd.Series(y_train_processed).value_counts(normalize=True).round(3)}"
    )
    print(f"Number of final features: {len(final_feature_names)}")

    # --- 5. Save Processed Data ---
    output_dir = Path(output_directory)
    output_dir.mkdir(exist_ok=True)

    train_final_df = pd.DataFrame(X_train_processed, columns=final_feature_names)
    train_final_df["Status"] = y_train_processed

    test_final_df = pd.DataFrame(X_test_processed, columns=final_feature_names)
    test_final_df["Status"] = y_test.reset_index(drop=True)

    train_final_df.to_csv(output_dir / "train_processed.csv", index=False)
    test_final_df.to_csv(output_dir / "test_processed.csv", index=False)

    print(f"Processed data saved to '{output_dir}/'.")
    print(
        f"Train data saved: '{output_dir / 'train_processed.csv'}' ({train_final_df.shape})"
    )
    print(f"Test data saved: '{output_dir / 'test_processed.csv'}' ({test_final_df.shape})")

if __name__ == "__main__":
    main(
        data_input_path="../data.csv", sep=";",
        output_directory="Graduate_indicators_preprocessing",
        ordinal_candidates=[],
        low_var_cols=[
            'Previous_qualification', 'Nacionality', 'Curricular_units_1st_sem_credited',
            'Curricular_units_1st_sem_without_evaluations', 'Curricular_units_2nd_sem_credited'
        ]
    )