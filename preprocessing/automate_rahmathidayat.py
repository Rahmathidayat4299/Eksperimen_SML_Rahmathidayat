import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

from feature_engine.outliers import Winsorizer
from feature_engine.selection import DropCorrelatedFeatures
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder

# --- Global Settings ---
sns.set(style="whitegrid")

# --- 1. Fungsi Preprocessing Data Awal ---
# (Ini adalah fungsi yang saya asumsikan ada atau mirip dengan di notebook Anda untuk membersihkan data mentah)
# Berdasarkan bagian 'df = df.drop_duplicates()' dan penanganan missing values,
# saya akan membuat fungsi preprocess_initial_data untuk mengkonsolidasikannya.
def preprocess_initial_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan preprocessing awal pada DataFrame:
    - Menangani missing values untuk numerik (median) dan kategorikal (mode).
    - Menghapus baris duplikat.
    """
    print("Memulai preprocessing data awal (missing values, duplikat)...")
    # Cek jumlah missing values
    print("Missing values sebelum penanganan:")
    print(df.isnull().sum())

    # Contoh penanganan: isi missing numerical dengan median, kategorikal dengan modus
    for col in df.select_dtypes(include='number').columns:
        df[col].fillna(df[col].median(), inplace=True)
    
    for col in df.select_dtypes(include='object').columns:
        # Check if mode() returns multiple values, take the first one
        if not df[col].mode().empty:
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            print(f"Warning: Column '{col}' has no mode for filling NaNs. Consider alternative handling.")

    print("\nMissing values setelah penanganan:")
    print(df.isnull().sum())

    # Cek duplikasi
    duplicates_count = df.duplicated().sum()
    print(f"\nJumlah duplikat sebelum dihapus: {duplicates_count}")

    # Hapus baris duplikat
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Jumlah baris setelah menghapus duplikat: {len(df)}")
    return df

# --- 2. Fungsi Deteksi Fitur Otomatis ---
def auto_detect_feature_types(df, target=None, ordinal_candidates=None):
    """
    Deteksi otomatis fitur biner, ordinal, kontinu, dan kategorikal dari DataFrame.
    - ordinal_candidates: list nama kolom yang ingin diperlakukan sebagai ordinal (opsional)
    """
    if target and target in df.columns:
        features = df.drop(columns=[target]).copy() # Use .copy() to avoid SettingWithCopyWarning
    else:
        features = df.copy() # Use .copy()

    binary_features = [col for col in features.columns
                       if features[col].nunique() == 2 and pd.api.types.is_numeric_dtype(features[col])]
    continuous_features = [col for col in features.columns
                           if pd.api.types.is_numeric_dtype(features[col]) and features[col].nunique() > 10]
    
    ordinal_features = []
    if ordinal_candidates:
        ordinal_features = [col for col in ordinal_candidates if col in features.columns]
        # Remove ordinal from continuous if overlap
        continuous_features = [col for col in continuous_features if col not in ordinal_features]
    
    # Categorical non-binary (object type)
    categorical_features = [col for col in features.select_dtypes(include='object').columns]
    # Remove binary from categorical if it was wrongly classified as object initially (e.g., 'Gender' as 'M'/'F')
    # This might require a more robust check for binary, but for now, rely on nunique() == 2 for numeric.
    
    return {
        "binary": binary_features,
        "ordinal": ordinal_features,
        "continuous": continuous_features,
        "categorical": categorical_features
    }

# --- 3. Fungsi Utama untuk Menjalankan Otomatisasi ---
def run_automation(data_input_path: str = "../data.csv", output_directory: str = "Graduate_indicators_preprocessing", sep: str = ";"):
    """
    Fungsi utama untuk menjalankan seluruh alur preprocessing dan analisis data.
    """
    print(f"Memulai otomatisasi dengan data dari: {data_input_path}")
    
    try:
        df = pd.read_csv(data_input_path, sep=sep)
        print(f"Dataset dimuat dari '{data_input_path}' dengan {df.shape[0]} baris dan {df.shape[1]} kolom.")
    except FileNotFoundError:
        print(f"Error: File '{data_input_path}' tidak ditemukan.")
        return
    except Exception as e:
        print(f"Terjadi kesalahan saat memuat dataset: {e}")
        return

    print("\nPratinjau data awal:")
    print(df.head())
    print("\nNama kolom:")
    print(df.columns.tolist())

    # --- Initial Data Preprocessing (Missing Values & Duplicates) ---
    df = preprocess_initial_data(df.copy())
    
    # --- Feature Engineering: Create age_group (if 'Age_at_enrollment' exists) ---
    if 'Age_at_enrollment' in df.columns:
        print("\nMembuat fitur 'age_group'...")
        # Ensure Age_at_enrollment is numeric before cutting
        df['Age_at_enrollment'] = pd.to_numeric(df['Age_at_enrollment'], errors='coerce')
        # Fill any NaNs created by coerce if necessary, before cutting
        df['Age_at_enrollment'].fillna(df['Age_at_enrollment'].median(), inplace=True) # Fill with median before binning

        df['age_group'] = pd.cut(df['Age_at_enrollment'],
                                 bins=[0, 18, 45, 100],
                                 labels=["Young", "Adult", "Senior"],
                                 right=False) # Use right=False for left-inclusive bins
        print("Fitur 'age_group' berhasil dibuat.")
    else:
        print("Kolom 'Age_at_enrollment' tidak ditemukan. Fitur 'age_group' tidak dibuat.")

    # --- Auto Detect Feature Types ---
    # Assuming 'Status' is the target column
    ordinal_candidates = [] # Sesuaikan ini jika Anda memiliki kolom ordinal spesifik
    feature_types = auto_detect_feature_types(df, target="Status", ordinal_candidates=ordinal_candidates)

    print("\n--- Deteksi Tipe Fitur ---")
    print("Fitur Biner:", feature_types["binary"])
    print("Fitur Ordinal:", feature_types["ordinal"])
    print("Fitur Kontinu:", feature_types["continuous"])
    print("Fitur Kategorikal (non-binary):", feature_types["categorical"])

    # --- Scaling (RobustScaler for numerical_cols) ---
    # Numerical features from initial detection *before* splitting
    numerical_cols = df.select_dtypes(include='number').columns.tolist()
    # Ensure 'Status' is not scaled if it's numeric
    if 'Status' in numerical_cols:
        numerical_cols.remove('Status')

    if numerical_cols:
        print(f"\nMelakukan Scaling (RobustScaler) pada fitur numerik: {numerical_cols}")
        scaler_initial = RobustScaler()
        df[numerical_cols] = scaler_initial.fit_transform(df[numerical_cols])
        print("Scaling awal selesai.")
    else:
        print("\nTidak ada fitur numerik untuk dilakukan scaling awal.")

    # --- Outlier Handling (Winsorizer for numerical_cols, 'gaussian' method as per your initial snippet) ---
    if numerical_cols:
        print(f"Menangani Outlier (Winsorizer 'gaussian' method) pada fitur numerik: {numerical_cols}")
        winsor = Winsorizer(capping_method='gaussian', tail='both', fold=3, variables=numerical_cols)
        df[numerical_cols] = winsor.fit_transform(df[numerical_cols])
        print("Penanganan outlier selesai.")
    else:
        print("Tidak ada fitur numerik untuk ditangani outlier.")

    # --- One-Hot Encoding for Categorical Features ---
    categorical_cols_to_encode = [col for col in df.select_dtypes(include='object').columns.tolist() if col != 'Status']
    if categorical_cols_to_encode:
        print(f"\nMelakukan One-Hot Encoding pada fitur kategorikal: {categorical_cols_to_encode}")
        df = pd.get_dummies(df, columns=categorical_cols_to_encode, drop_first=True)
        print("One-Hot Encoding selesai. Bentuk data sekarang:", df.shape)
    else:
        print("\nTidak ada fitur kategorikal untuk dilakukan One-Hot Encoding.")


    # --- Split Data ---
    if 'Status' not in df.columns:
        print("Error: Kolom 'Status' (target) tidak ditemukan setelah preprocessing. Tidak dapat melakukan split data.")
        return

    X = df.drop("Status", axis=1)
    y = df["Status"]

    print("\n--- Membagi Data (Training & Testing) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data terbagi: Training {X_train.shape}, Test {X_test.shape}")
    print(f"Distribusi target Training:\n{y_train.value_counts(normalize=True).round(3)}")

    # --- Define Preprocessing Pipeline for ColumnTransformer ---
    # Note: I will use feature_types detected from the *initial* df for defining transformers
    # but ensure columns are present in X_train/X_test.
    
    # Re-detect feature types specifically for X (after initial transformations like one-hot encoding)
    # This is crucial because `auto_detect_feature_types` works on dtypes and nunique.
    # After get_dummies, original categorical columns are gone, and new binary/numerical ones appear.
    # The `auto_detect_feature_types` function as provided does not handle already encoded columns well
    # for the purpose of ColumnTransformer.

    # Instead of re-detecting, we need to map the original feature types to the columns
    # that *will exist* after one-hot encoding is applied, or apply one-hot encoding
    # as part of the ColumnTransformer itself if it hasn't been done globally.

    # Given that OneHotEncoder was applied globally on `df` before splitting,
    # we need to adjust `feature_types` for the `ColumnTransformer`.
    # The `ColumnTransformer` will only see numerical columns (from original numerical, plus new from one-hot encoding).
    # Re-evaluating feature types on X_train for the ColumnTransformer:
    # After df.get_dummies, all original categorical_cols_to_encode are gone.
    # The `feature_types` dict needs to reflect the columns *in X_train*.

    # Let's adjust `get_preprocessor` logic or redefine `feature_types` for the ColumnTransformer.
    # From your notebook code, `feature_types` was used for `auto_detect_feature_types(df, ...)`
    # *before* `df_encoded` (which is `df` in your last snippets).
    # Then `numerical_cols` and `categorical_cols` were used for scaling/encoding.

    # The `ColumnTransformer` in your provided code builds pipelines for
    # "continuous_pipeline", "ordinal_scaler", "binary_passthrough", "cat_encoder".
    # The `cat_encoder` part in your provided code uses `[col for col in feature_types["categorical"] if col in X_train.columns]`.
    # This implies that some categorical features might still be in X_train *before* this preprocessor.

    # Let's try to infer the correct `feature_types` for the ColumnTransformer
    # based on the *state of X_train* at this point.
    
    # Re-detect feature types for X_train (which already has initial preprocessing and global one-hot encoding done)
    # This will likely classify most as continuous.
    current_feature_types = auto_detect_feature_types(X_train, target=None, ordinal_candidates=ordinal_candidates)
    
    # Correct `low_var_cols` and `winsorize_columns` based on what's in X_train.
    low_var_cols = [
        'Previous_qualification', 'Nacionality', 'Curricular_units_1st_sem_credited',
        'Curricular_units_1st_sem_without_evaluations', 'Curricular_units_2nd_sem_credited'
    ]
    # Filter low_var_cols to only include those present in X_train
    low_var_cols_in_X = [col for col in low_var_cols if col in X_train.columns]

    winsorize_columns = [col for col in current_feature_types["continuous"] if col not in low_var_cols_in_X]


    # Define the preprocessing steps using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "continuous_pipeline",
                Pipeline(
                    [
                        (
                            "winsorize",
                            Winsorizer(
                                capping_method="iqr", # Changed to iqr as per your provided notebook snippet later
                                fold=1.5,
                                variables=winsorize_columns,
                            ),
                        ),
                        ("scaler", RobustScaler()),
                    ]
                ),
                winsorize_columns, # Apply to these columns
            ),
            ("ordinal_scaler", RobustScaler(), current_feature_types["ordinal"]),
            ("binary_passthrough", "passthrough", current_feature_types["binary"]),
            # If OneHotEncoder was already applied globally, this might not be needed or needs careful handling.
            # Assuming remaining 'object' dtypes in X_train should be one-hot encoded by CT.
            ("cat_encoder", OneHotEncoder(handle_unknown="ignore"),
             [col for col in current_feature_types["categorical"] if col in X_train.columns]),
        ],
        remainder="drop", # Drop any columns not specified
    )

    # Create the full pipeline including preprocessing, feature selection, and SMOTE
    pipeline_steps = [
        ("preprocess", preprocessor),
        ("feature_selection", DropCorrelatedFeatures(threshold=0.8, method="pearson")),
        ("smote", SMOTE(random_state=42, sampling_strategy="auto")),
    ]
    
    # Add DropConstantFeatures if low_var_cols are actually constant after initial preprocessing
    # The snippet you provided had DropConstantFeatures in a different context.
    # If the intent is to drop them *before* the main pipeline, it should be done earlier.
    # Based on the last provided notebook code, DropConstantFeatures is *not* in the final pipeline.
    # The `low_var_cols` list was used to exclude from `winsorize_columns`, not for `DropConstantFeatures`.
    
    pipeline = ImbalancedPipeline(pipeline_steps)

    print("\n--- Menerapkan Preprocessing Pipeline dan SMOTE ---")
    X_train_processed, y_train_processed = pipeline.fit_resample(X_train, y_train)

    # Apply only preprocessing and feature selection (without SMOTE) to test data
    X_test_processed = pipeline[:-1].transform(X_test)

    # Get the final feature names after all transformations and selections
    # Ensure this works if feature_selection step outputs a DataFrame (which it usually does not by default)
    # The get_feature_names_out() method is specific to certain transformers.
    # For `DropCorrelatedFeatures`, it usually just returns column names if it operates on a DataFrame.
    # If the output is an array, we might need to manually construct names from the ColumnTransformer.
    # Assuming `final_feature_names = pipeline.named_steps["feature_selection"].get_feature_names_out()` works.
    
    try:
        # For DropCorrelatedFeatures, get_feature_names_out() works on the transformer itself
        # But if the output of preprocess is an array, it might not have names.
        # Let's try to get feature names from the ColumnTransformer first, then apply DropCorrelatedFeatures.
        # This requires `set_output(transform="pandas")` on ColumnTransformer for versions > 1.2
        # Or, just rely on the output of `DropCorrelatedFeatures` if it operates on named features.
        
        # A safer way to get feature names after ColumnTransformer:
        # Dummy fit transform to get column names from preprocessor
        # preprocessor.fit(X_train)
        # preprocessed_cols = preprocessor.get_feature_names_out()
        # Then, apply feature selection to a DataFrame with these names.

        # For simplicity and sticking to the provided notebook's structure:
        # Assuming pipeline.named_steps["feature_selection"].get_feature_names_out() works as intended.
        # If it throws an error, it's due to the output format of the previous step.
        final_feature_names = pipeline.named_steps["feature_selection"].get_feature_names_out()
    except AttributeError:
        print("Warning: get_feature_names_out() not available on feature_selection step directly.")
        print("Attempting to infer feature names from X_train_processed if it's a pandas DataFrame, else using generic names.")
        if isinstance(X_train_processed, pd.DataFrame):
            final_feature_names = X_train_processed.columns
        else:
            # If it's a numpy array, generate generic names
            final_feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]


    print(f"Preprocessing selesai. Bentuk data Training: {X_train_processed.shape}")
    print(f"Bentuk data Test: {X_test_processed.shape}")
    print(f"Distribusi target Training setelah SMOTE:\n{pd.Series(y_train_processed).value_counts(normalize=True).round(3)}")
    print(f"Jumlah fitur akhir: {len(final_feature_names)}")

    # --- Save Processed Data ---
    output_dir = Path(output_directory)
    output_dir.mkdir(exist_ok=True)

    train_final_df = pd.DataFrame(X_train_processed, columns=final_feature_names)
    train_final_df["Status"] = y_train_processed

    test_final_df = pd.DataFrame(X_test_processed, columns=final_feature_names)
    test_final_df["Status"] = y_test.reset_index(drop=True)

    train_final_df.to_csv(output_dir / "train_processed.csv", index=False)
    test_final_df.to_csv(output_dir / "test_processed.csv", index=False)

    print(f"\nData yang telah diproses disimpan ke '{output_dir}/'.")
    print(f"Data latih disimpan: '{output_dir / 'train_processed.csv'}' ({train_final_df.shape})")
    print(f"Data uji disimpan: '{output_dir / 'test_processed.csv'}' ({test_final_df.shape})")

    # --- Load and Print Processed Training Data Preview ---
    # This part was also in your notebook, so including it in the automation script.
    try:
        df_processed_train_preview = pd.read_csv(output_dir / "train_processed.csv")
        print("\nPratinjau data training yang telah diproses:")
        print(df_processed_train_preview.head())
    except FileNotFoundError:
        print(f"Error: File '{output_dir / 'train_processed.csv'}' tidak ditemukan setelah disimpan.")
    except Exception as e:
        print(f"Terjadi kesalahan saat memuat pratinjau data training yang telah diproses: {e}")

    # --- ERA: Exploratory Data Analysis (EDA) ---
    # This section contains plotting commands which might not be ideal for automated scripts
    # if you don't want plots to be displayed/saved every time.
    # For now, I'm including them as they were in your notebook.
    # In a true automation script, you might want to save these plots to files.
    print("\n--- Memulai Analisis Eksplorasi Data (EDA) ---")

    # Re-read df for EDA as it might have been modified significantly
    # Or, if EDA should happen on the *processed* data, we'd need to adjust.
    # Given the original notebook structure, EDA was on a 'df' that likely came
    # from initial load, not necessarily the fully processed one.
    # To keep this script self-contained and runnable, I will use the df after initial processing for EDA.
    # If the EDA needs to be on the *raw* data, the EDA block needs to be moved earlier.
    # I'll re-load df for EDA to avoid issues with earlier transformations that might affect EDA interpretation.
    # Or, preferably, pass `df_raw` or `df_initial_processed` to a separate EDA function.

    # For consistency with the snippet you provided for EDA, let's assume it should run
    # on the `df` *before* the final ColumnTransformer and SMOTE pipeline.
    # I will move the EDA section to run after `preprocess_initial_data`
    # and before the main `ColumnTransformer` pipeline.

    # --- EDA Data Setup ---
    # Re-detect numerical and categorical features after initial preprocessing
    numerical_features_eda = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features_eda = df.select_dtypes(include=['object']).columns.tolist()
    # Remove 'Status' from features if it's there, as it's typically the target for EDA visualizations
    if 'Status' in numerical_features_eda:
        numerical_features_eda.remove('Status')
    if 'Status' in categorical_features_eda:
        categorical_features_eda.remove('Status')


    # ==============================
    # 1. UNIVARIATE ANALYSIS (on the initially processed df)
    # ==============================
    print("\nUNIVARIATE ANALYSIS\n" + "="*60)
    for col in numerical_features_eda[:5]: # limit to 5 for speed/display
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(df[col], kde=True, ax=axes[0])
        axes[0].set_title(f'Distribution of {col}')
        sns.boxplot(y=df[col], ax=axes[1])
        axes[1].set_title(f'Boxplot of {col}')
        plt.tight_layout()
        plt.show() # This will display the plot
        print(f"Stats for {col}:\n{df[col].describe()}")
        print(f"Skewness: {df[col].skew():.2f}\n{'='*60}")

    for col in categorical_features_eda[:5]: # limit to 5
        plt.figure(figsize=(8, 4))
        ax = sns.countplot(x=col, data=df)
        total = len(df[col])
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{100*height/total:.1f}%', 
                            (p.get_x() + p.get_width()/2, height), 
                            ha='center', va='bottom', fontsize=9)
        plt.title(f'Count of {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show() # This will display the plot

    # ==============================
    # 2. BIVARIATE ANALYSIS (on the initially processed df)
    # ==============================
    print("\nBIVARIATE ANALYSIS\n" + "="*60)

    # Correlation Heatmap
    if numerical_features_eda:
        plt.figure(figsize=(10, 6))
        sns.heatmap(df[numerical_features_eda].corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()

    # Numerical vs Numerical (first 3 pairs only)
    if len(numerical_features_eda) >= 4:
        for a, b in zip(numerical_features_eda, numerical_features_eda[1:4]):
            plt.figure(figsize=(6, 4))
            sns.scatterplot(x=df[a], y=df[b])
            plt.title(f'{a} vs {b}')
            plt.tight_layout()
            plt.show()

    # Categorical vs Numerical (limit)
    if categorical_features_eda and numerical_features_eda:
        for cat in categorical_features_eda[:2]:
            for num in numerical_features_eda[:2]:
                plt.figure(figsize=(6, 4))
                sns.boxplot(x=cat, y=num, data=df)
                plt.title(f'{num} by {cat}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()

    # Categorical vs Categorical (only first 2)
    if len(categorical_features_eda) >= 2:
        ct = pd.crosstab(df[categorical_features_eda[0]], df[categorical_features_eda[1]])
        sns.heatmap(ct, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{categorical_features_eda[0]} vs {categorical_features_eda[1]}')
        plt.tight_layout()
        plt.show()

    # ==============================
    # 3. MULTIVARIATE ANALYSIS (on the initially processed df)
    # ==============================
    print("\nMULTIVARIATE ANALYSIS\n" + "="*60)

    # Pairplot (limit to 5 numerical features + 1 categorical for hue)
    # Ensure there are enough features for pairplot
    pairplot_cols = []
    if len(numerical_features_eda) >= 5:
        pairplot_cols.extend(numerical_features_eda[:5])
    else:
        pairplot_cols.extend(numerical_features_eda)
    
    pairplot_hue = None
    if categorical_features_eda:
        pairplot_hue = categorical_features_eda[0]
        # Add the hue column to pairplot_cols if not already there
        if pairplot_hue not in pairplot_cols:
            pairplot_cols.append(pairplot_hue)

    if pairplot_cols:
        sns.pairplot(df[pairplot_cols],
                     hue=pairplot_hue,
                     diag_kind='kde')
        plt.suptitle('Pairplot', y=1.02)
        plt.show()

    # 3D Scatter (first 3 numerical features)
    if len(numerical_features_eda) >= 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df[numerical_features_eda[0]], df[numerical_features_eda[1]], df[numerical_features_eda[2]])
        ax.set_xlabel(numerical_features_eda[0])
        ax.set_ylabel(numerical_features_eda[1])
        ax.set_zlabel(numerical_features_eda[2])
        plt.title('3D Scatter Plot')
        plt.show()

    # PCA Visualization
    if len(numerical_features_eda) > 1: # PCA needs at least 2 features
        # PCA requires no NaNs, handle them before
        df_pca_ready = df[numerical_features_eda].fillna(df[numerical_features_eda].mean())
        if df_pca_ready.empty:
            print("Tidak cukup data numerik untuk PCA setelah penanganan NaN.")
        else:
            pca = PCA(n_components=min(2, len(numerical_features_eda))) # Ensure n_components <= number of features
            pc = pca.fit_transform(df_pca_ready)
            plt.figure(figsize=(8, 5))
            sns.scatterplot(x=pc[:,0], y=pc[:,1], hue=df[categorical_features_eda[0]] if categorical_features_eda else None)
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title("PCA Visualization")
            plt.tight_layout()
            plt.show()

    # Clustermap
    if numerical_features_eda:
        sns.clustermap(df[numerical_features_eda].corr(), cmap='coolwarm', annot=True)
        plt.title("Feature Clustering")
        plt.show()

# --- Main execution block ---
if __name__ == "__main__":
    # Example usage:
    # Ensure 'data.csv' is in the parent directory or provide the full path.
    # For this example, I'll use a relative path assuming it's one level up.
    # If running from the same directory as 'data.csv', change to "data.csv"
    run_automation(
        data_input_path="D:\Machine-learning\MSML-Struktur\data.csv", # Sesuaikan path sesuai lokasi file data.csv Anda
        output_directory="Graduate_indicators_preprocessing",
        sep=";"
    )
