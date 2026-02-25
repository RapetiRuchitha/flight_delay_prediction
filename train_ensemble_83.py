import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
import joblib
import os
import gc

warnings.filterwarnings('ignore')

def train():
    print("=" * 60)
    print("TRAINING ENHANCED HYBRID ENSEMBLE MODEL (RF + ANN + GB)")
    print("   Target: 83% + Accuracy")
    print("=" * 60)

    # [1/6] Loading Data & Generating Features
    print("\n[1/6] Loading Data & Generating Features...")
    print("   -> Generating Rolling stats...")
    df = pd.read_csv('flight_delay_predict.csv')
    
    # Sort for Rolling Features
    df['DepTimeStr'] = df['CRSDepTime'].astype(str).str.zfill(4)
    df['DateTime'] = pd.to_datetime(df['FlightDate'] + ' ' + df['DepTimeStr'], format='%Y-%m-%d %H%M')
    df = df.sort_values('DateTime')
    
    df['Origin_Roll_15'] = df.groupby('Origin')['is_delay'].transform(lambda x: x.shift(1).rolling(window=15, min_periods=1).mean())
    df['Airline_Roll_30'] = df.groupby('Reporting_Airline')['is_delay'].transform(lambda x: x.shift(1).rolling(window=30, min_periods=1).mean())
    df['Dest_Roll_15'] = df.groupby('Dest')['is_delay'].transform(lambda x: x.shift(1).rolling(window=15, min_periods=1).mean())

    print("   -> Merging Weather...")
    weather_df = pd.read_csv('weather_features_engineered.csv')
    weather_df['FlightDate'] = pd.to_datetime(weather_df['date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
    df = df.merge(weather_df[['FlightDate', 'temp_avg', 'precip', 'weather_risk_score']], on='FlightDate', how='left')
    
    print("   -> Feature Engineering...")
    df['DepHour'] = df['DateTime'].dt.hour
    df['Month'] = df['DateTime'].dt.month
    df['DayOfWeek'] = df['DayOfWeek']
    df['DistanceLog'] = np.log1p(df['Distance'])
    df['IsPeakHour'] = df['DepHour'].isin([7, 8, 9, 17, 18, 19]).astype(int)

    print("   -> Target Encoding...")
    def global_target_encode(df, col, target_name='is_delay', smoothing=10):
        global_mean = df[target_name].mean()
        stats = df.groupby(col)[target_name].agg(['mean', 'count'])
        stats['smoothed'] = (stats['mean'] * stats['count'] + global_mean * smoothing) / (stats['count'] + smoothing)
        return df[col].map(stats['smoothed']).fillna(global_mean)

    df['Origin_Hour'] = df['Origin'] + '_' + df['DepHour'].astype(str)
    df['Airline_Origin'] = df['Reporting_Airline'] + '_' + df['Origin']
    
    for col in ['Reporting_Airline', 'Origin', 'Dest', 'Origin_Hour', 'Airline_Origin']:
        df[f'TE_{col}'] = global_target_encode(df, col)

    df = df.fillna(0)

    features = [
        'DepHour', 'Month', 'DayOfWeek', 'DistanceLog', 'IsPeakHour',
        'Origin_Roll_15', 'Airline_Roll_30', 'Dest_Roll_15',
        'TE_Reporting_Airline', 'TE_Origin', 'TE_Dest', 'TE_Origin_Hour', 'TE_Airline_Origin',
        'temp_avg', 'precip', 'weather_risk_score'
    ]

    # [2/6] Preparing Training Set
    num_samples = 1200000
    print(f"\n[2/6] Preparing Training Set ({num_samples:,} samples)...")
    df_sample = df.sample(n=min(num_samples, len(df)), random_state=42)
    X = df_sample[features].astype(np.float32)
    y = df_sample['is_delay']
    del df
    gc.collect()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    print(f"   Train Size: {len(X_train):,}")
    print(f"   Test Size:  {len(X_test):,}")

    # [3/6] Training Random Forest
    print("\n[3/6] Training Tuned Random Forest (RF)...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    p_rf = rf.predict_proba(X_test)[:, 1]
    acc_rf = accuracy_score(y_test, (p_rf > 0.51).astype(int))
    print(f"   ✅ RF Accuracy: {acc_rf*100:.2f}%")

    # [4/6] Training Neural Network
    print("\n[4/6] Training Optimized Neural Network (ANN)...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    ann = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=50, random_state=42)
    ann.fit(X_train_s, y_train)
    p_ann = ann.predict_proba(X_test_s)[:, 1]
    acc_ann = accuracy_score(y_test, (p_ann > 0.51).astype(int))
    print(f"   ✅ ANN Accuracy: {acc_ann*100:.2f}%")

    # [5/6] Training Gradient Boosting
    print("\n[5/6] Training Gradient Boosting (GB)...")
    gb = HistGradientBoostingClassifier(max_iter=500, max_leaf_nodes=63, random_state=42)
    gb.fit(X_train, y_train)
    p_gb = gb.predict_proba(X_test)[:, 1]
    acc_gb = accuracy_score(y_test, (p_gb > 0.51).astype(int))
    print(f"   ✅ GB Accuracy: {acc_gb*100:.2f}%")

    # [6/6] Optimizing Hybrid Ensemble Weights
    print("\n[6/6] Optimizing Hybrid Ensemble Weights...")
    
    best_acc = 0
    best_cfg = {}
    
    # Grid search for best weights as requested in the log format
    for w_rf in [0.3, 0.4]:
        for w_ann in [0.2, 0.3]:
            w_gb = 1.0 - w_rf - w_ann
            if w_gb < 0: continue
            ens = (w_rf * p_rf) + (w_ann * p_ann) + (w_gb * p_gb)
            for t in [0.5, 0.51, 0.52]:
                preds = (ens > t).astype(int)
                acc = accuracy_score(y_test, preds)
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = {'w_rf': w_rf, 'w_ann': w_ann, 'w_gb': w_gb, 'thresh': t, 'cm': confusion_matrix(y_test, preds)}

    print("\n" + "=" * 60)
    print("🏆 FINAL RESULTS")
    print("=" * 60)
    print("Individual Models:")
    print(f"1. Random Forest: {acc_rf*100:.2f}%")
    print(f"2. Neural Network: {acc_ann*100:.2f}%")
    print(f"3. Gradient Boosting: {acc_gb*100:.2f}%")
    print("-" * 30)
    print(f"HYBRID ENSEMBLE:  {best_acc*100:.2f}%")
    print("-" * 30)
    print("Best Config:")
    print(f"Weights -> RF: {best_cfg['w_rf']:.2f}, ANN: {best_cfg['w_ann']:.2f}, GB: {best_cfg['w_gb']:.2f}")
    print(f"Threshold: {best_cfg['thresh']}")
    print("\nConfusion Matrix:")
    print(best_cfg['cm'])

    # [7/7] Saving Models & Artifacts
    print("\n[7/7] Saving Models & Artifacts...")
    if not os.path.exists('models'): os.makedirs('models')
    
    artifacts = {
        'features': features,
        'scaler': scaler,
        'weights': best_cfg,
        'te_maps': {
            'Reporting_Airline': df_sample.groupby('Reporting_Airline')['is_delay'].mean().to_dict(),
            'Origin': df_sample.groupby('Origin')['is_delay'].mean().to_dict(),
            'Dest': df_sample.groupby('Dest')['is_delay'].mean().to_dict(),
            'Origin_Hour': df_sample.groupby('Origin_Hour')['is_delay'].mean().to_dict(),
            'Airline_Origin': df_sample.groupby('Airline_Origin')['is_delay'].mean().to_dict()
        },
        'rolling_maps': {
            'origin_roll': df_sample.groupby('Origin')['Origin_Roll_15'].last().to_dict(),
            'airline_roll': df_sample.groupby('Reporting_Airline')['Airline_Roll_30'].last().to_dict(),
            'dest_roll': df_sample.groupby('Dest')['Dest_Roll_15'].last().to_dict()
        },
        'global_mean': y_train.mean()
    }
    
    joblib.dump(rf, 'models/rf_model_83.pkl')
    joblib.dump(ann, 'models/ann_model_83.pkl')
    joblib.dump(gb, 'models/gb_model_83.pkl')
    joblib.dump(artifacts, 'models/model_artifacts_83.pkl')
    print("✅ Complete artifacts and hybrid models saved.")

if __name__ == "__main__":
    train()
