import os
from functools import partial
from kfp.components import create_component_from_func
from kfp import onprem
import argparse
import kfp
import requests
from dotenv import load_dotenv, dotenv_values

dotenv_path = '/mnt/dlabflow/structured/config'
load_dotenv(dotenv_path)

KubeflowPieplineTraining = os.getenv('KubeflowPieplineTraining')
KubeflowHost = os.getenv('KubeflowHost')
KubeflowUsername = os.getenv('KubeflowUsername1')
KubeflowPassword = os.getenv('KubeflowPassword1')
KubeflowNamespace = os.getenv('KubeflowNamespace1')
KubeflowVolumeName = os.getenv('KubeflowVolumeName1')
KubeflowVolumeMountPath = os.getenv('KubeflowVolumeMountPath1')

@partial(create_component_from_func, base_image='dgkim1983/dlabflow:structured-20251016-latest')
def Training(projectId: str, versionId: str, target: str, feature: str, algorithm: str, tuning: bool, advancedSettingForClassification: str, status: str, statusMessage: str, trainingResult: str):
    import os
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder, label_binarize
    from minio import Minio
    import requests
    from dotenv import load_dotenv, dotenv_values
    import json
    import traceback
    import logging
    import redis
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, make_scorer
    import plotly.graph_objects as go
    import lightgbm as lgb
    from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    import base64
    import joblib
    from joblib import dump, load
    import pickle
    import zipfile
    import optuna

    logger = logging.getLogger('Kubeflow Log')
    if logger.hasHandlers():
        logger.handlers.clear()
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    dotenv_path = '/mnt/dlabflow/structured/config'
    load_dotenv(dotenv_path)

    RedisHost = os.getenv('RedisHost')
    RedisPort = os.getenv('RedisPort')
    RedisDB = os.getenv('RedisDB')
    r = redis.Redis(host=RedisHost, port=RedisPort, db=RedisDB)

    MinIODefaultPath = os.getenv('MinIODefaultPath')
    MinioEndpoint = os.getenv('MinioEndpoint')
    MinioAccessKey = os.getenv('MinioAccessKey')
    MinioSecretKey = os.getenv('MinioSecretKey')
    MinioSecure = os.getenv('MinioSecure', 'False').lower() == 'true'
    MinioBucketUser = os.getenv('MinioBucketUser1')
    MinioBucketPreprocessing = os.getenv('MinioBucketPreprocessing')
    MinIOBucketTraining = os.getenv('MinIOBucketTraining')
    BentomlTrainingGet = os.getenv('BentomlTrainingGet')

    preprocessing_key = f"preprocessing:{projectId}:{versionId}"
    preprocessing_data_json = r.get(preprocessing_key)
    if preprocessing_data_json:
        preprocessing_data = json.loads(preprocessing_data_json)
        train_ratio = preprocessing_data.get('ratioOfTraining')
        val_ratio = preprocessing_data.get('ratioOfValidation')
        test_ratio = preprocessing_data.get('ratioOfTesting')

    base_paths = f"/{MinIODefaultPath}/{MinioBucketUser}/{projectId}/{versionId}/{MinIOBucketTraining}/"
    preprocessing_path = f"/{MinIODefaultPath}/{MinioBucketUser}/{projectId}/{versionId}/{MinioBucketPreprocessing}/preprocessing.csv"
    os.makedirs(base_paths, exist_ok=True)

    client = Minio(
        endpoint=MinioEndpoint,
        access_key=MinioAccessKey,
        secret_key=MinioSecretKey,
        secure=MinioSecure
    )

    def send_status_to_bentoml(projectId, versionId, algorithm, status, statusMessage=None, trainingResult=None, progress=None, accuracy=None, precision=None, recall=None, auc=None):
        try:
            url = BentomlTrainingGet
            payload = {
                'projectId': projectId,
                'versionId': versionId,
                'status': status,
            }
            if algorithm:
                payload['algorithm'] = algorithm
            if statusMessage:
                payload['statusMessage'] = statusMessage
            if trainingResult is not None:
                payload['trainingResult'] = trainingResult
            if progress is not None:
                payload['progress'] = progress
            if accuracy is not None:
                payload['accuracy'] = accuracy
            if precision is not None:
                payload['precision'] = precision
            if recall is not None:
                payload['recall'] = recall
            if auc is not None:
                payload['auc'] = auc            
            requests.post(url, json=payload, timeout=3)
        except Exception:
            statusMessage = '학습 결과를 전송하는 과정에서 오류가 발생하였습니다.'
            logger.error(f"{statusMessage}")
            send_status_to_bentoml(projectId, versionId, status='ERROR', statusMessage=statusMessage)
            raise

    def csv_reader(data_path, chunksize):
        try:
            reader = pd.read_csv(data_path, chunksize=chunksize)
            chunks = []
            chunk_count = 0
            for i, chunk in enumerate(reader, 1):
                chunks.append(chunk)
                chunk_count += 1
            data = pd.concat(chunks, ignore_index=True)
            return data
        except Exception:
            statusMessage = '학습 데이터를 불러오는 과정에서 오류가 발생하였습니다.'
            logger.error(f"{statusMessage}")
            send_status_to_bentoml(projectId, versionId, algorithm, status='ERROR', statusMessage=statusMessage)
            raise

    def map_features_to_df_columns(df, feature_list):
        try:
            mapped_features = []
            for feat in feature_list:
                matched_cols = [col for col in df.columns if col.startswith(feat)]
                if matched_cols:
                    mapped_features.extend(matched_cols)
                else:
                    mapped_features.append(feat)
            return mapped_features
        except Exception:
            statusMessage = '수치화된 범주형 데이터를 원래 범주형 값으로 복원하는 과정에서 오류가 발생하였습니다.'
            logger.error(f"{statusMessage}")
            send_status_to_bentoml(projectId, versionId, algorithm, status='ERROR', statusMessage=statusMessage)
            raise        

    def is_continuous(series, unique_threshold=10):
        if pd.api.types.is_numeric_dtype(series):
            if series.nunique() > unique_threshold:
                statusMessage = '종속 변수가 연속형 값으로 회귀에 적합하며, 현재 학습 모델에서 회귀는 지원하지 않습니다.'
                logger.error(f"{statusMessage}")
                send_status_to_bentoml(projectId, versionId, algorithm, status='ERROR', statusMessage=statusMessage)
                raise
            else:
                return False
        return False

    def train_and_plot(
        algorithm='lightgbm',
        #tuning=False,
        #optimizationMetric='accuracy',
        #optimizationAlgorithm=None, 
        #learningRateScheduler=False, 
        #minimumLearningRate=None, 
        #maximumLearningRate=None, 
        #optimizationIterations=None, 
        #earlyStop=None, 
        n_estimators=50, 
        show=False,
        chart_save=False,
        #random_state=42            
    ):
        df = csv_reader(data_path=preprocessing_path, chunksize=10000)
        actual_feature_columns = map_features_to_df_columns(df, feature)
        if df[target].dtype == 'object':
            statusMessage = f"종속 변수 {target}(이)가 수치형으로 변환되지 않아 학습 모델에 사용할 수 없습니다."
            logger.error(f"{statusMessage}")
            send_status_to_bentoml(projectId, versionId, algorithm, status='ERROR', statusMessage=statusMessage)
            raise

        num_missing = df[target].isnull().sum()
        total = len(df)
        if num_missing > 0:
            missing_ratio = num_missing / total
            if missing_ratio >= 0.1:
                statusMessage = f"종속 변수에 결측값이 전체의 {missing_ratio:.2%}로 10% 이상 존재하여, 학습 모델에 사용할 수 없습니다. 버전 생성 단계에서 해당 변수의 결측치를 처리하는 작업이 필요합니다."
                logger.error(f"{statusMessage}")
                send_status_to_bentoml(projectId, versionId, algorithm, status='ERROR', statusMessage=statusMessage)
                raise
            else:
                df = df.dropna(subset=[target])

        is_continuous(df[target])
        
        unique_labels = pd.unique(df[target])
        num_classes = len(unique_labels)
        if num_classes < 2:
            statusMessage = f"종속 변수 값이 2개 이상이어야 학습 모델에 사용할 수 있습니다."
            logger.error(f"{statusMessage}")
            send_status_to_bentoml(projectId, versionId, algorithm, status='ERROR', statusMessage=statusMessage)
            raise            

        """
        categorical_columns = df[actual_feature_columns].select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_columns:
            statusMessage = f"독립 변수 {categorical_columns}에 범주형 데이터가 존재하여, 학습 모델에 사용할 수 없습니다. 버전 생성 단계에서 해당 변수를 수치화하는 작업이 필요합니다."
            logger.error(f"{statusMessage}")
            send_status_to_bentoml(projectId, versionId, algorithm, status='ERROR', statusMessage=statusMessage)
            raise
        """

        missing_columns = df[actual_feature_columns].columns[df[actual_feature_columns].isnull().any()].tolist()
        if missing_columns:
            statusMessage = f"독립 변수 {missing_columns}에 결측값이 존재하여, 학습 모델에 사용할 수 없습니다. 버전 생성 단계에서 해당 변수의 결측치를 처리하는 작업이 필요합니다."
            logger.error(f"{statusMessage}")
            send_status_to_bentoml(projectId, versionId, algorithm, status='ERROR', statusMessage=statusMessage)
            raise

        df_total_len = len(df)
        train_count = round(df_total_len * train_ratio/100)
        val_count = round(df_total_len * val_ratio/100)
        test_count = df_total_len - train_count - val_count

        X = df[actual_feature_columns].values
        y = df[target].values

        #X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=(1 - train_ratio/100), random_state=42)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_count, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=val_count, random_state=42)

        train_acc_list = []
        val_acc_list = []
        algo = algorithm.lower()

        if algo == 'lightgbm':
            model = lgb.LGBMClassifier(
                objective='multiclass' if num_classes > 2 else 'binary',
                num_class=num_classes if num_classes > 2 else None,
                n_estimators=n_estimators,
                random_state=42,
                verbose=-1
            )
            def lgb_accuracy(y_true, y_pred):
                if num_classes == 2:
                    y_pred_labels = (y_pred > 0.5).astype(int)
                else:
                    y_pred_labels = np.argmax(y_pred, axis=1)
                return 'accuracy', np.mean(y_true == y_pred_labels), True

            def record_accuracy(env):
                for data_name, eval_name, result, _ in env.evaluation_result_list:
                    if eval_name == 'accuracy':
                        if data_name == 'training':
                            train_acc_list.append(result)
                        elif data_name == 'valid_0':
                            val_acc_list.append(result)

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val), (X_train, y_train)],
                eval_metric=lgb_accuracy,
                callbacks=[record_accuracy, lgb.log_evaluation(period=0)]
            )

        elif algo == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                random_state=42,
                loss='log_loss'
            )
            model.fit(X_train, y_train)
            for train_pred, val_pred in zip(model.staged_predict(X_train), model.staged_predict(X_val)):
                train_acc_list.append(accuracy_score(y_train, train_pred))
                val_acc_list.append(accuracy_score(y_val, val_pred))

        elif algo == 'ada_boost':
            base = DecisionTreeClassifier(max_depth=1, random_state=42)
            model = AdaBoostClassifier(
                estimator=base,
                n_estimators=n_estimators,
                random_state=42,
                algorithm='SAMME'
            )
            model.fit(X_train, y_train)
            for train_pred, val_pred in zip(model.staged_predict(X_train), model.staged_predict(X_val)):
                train_acc_list.append(accuracy_score(y_train, train_pred))
                val_acc_list.append(accuracy_score(y_val, val_pred))

        y_train_pred = model.predict(X_train)
        if hasattr(model, "predict_proba"):
            y_train_prob = model.predict_proba(X_train)
            if num_classes == 2:
                y_train_prob = y_train_prob[:, 1]
        else:
            y_train_prob = y_train_pred

        metrics_score = {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
        }

        if num_classes == 2:
            metrics_score['auc'] = roc_auc_score(y_train, y_train_prob)
        else:
            y_train_bin = label_binarize(y_train, classes=unique_labels)
            metrics_score['auc'] = roc_auc_score(y_train_bin, y_train_prob, multi_class='ovr', average='weighted')

        title_text = algorithm.replace('_', ' ').title()
        iterations = list(range(1, len(train_acc_list) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=iterations, y=train_acc_list, mode='lines', name='Train'))
        fig.add_trace(go.Scatter(x=iterations, y=val_acc_list, mode='lines', name='Validation'))
        y_min = min(min(train_acc_list), min(val_acc_list)) - 0.02
        y_max = max(max(train_acc_list), max(val_acc_list)) + 0.02
        fig.update_layout(
            title=f"{title_text} Training & Validation Accuracy",
            xaxis_title="Iteration",
            yaxis_title="Accuracy",
            yaxis=dict(range=[y_min, y_max]),
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=700
        )
        if show:
            fig.show()
        if chart_save:
            fig.write_image(base_paths+'metirc_chart.png')
        fi_fig = None
        feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'Feature {i}' for i in range(X_train.shape[1])]
        importances = model.feature_importances_
        importances = importances / importances.sum()
        fi_fig = go.Figure()
        fi_fig.add_trace(go.Bar(x=importances, y=actual_feature_columns, orientation='h', marker_color='indianred'))
        height_px = max(400, 50 * len(feature_names))
        fi_fig.update_layout(
            title = f"{title_text} Feature Importance",
            xaxis_title = 'Contribution to Model Prediction',
            yaxis_title = 'Features',
            plot_bgcolor = 'white',
            paper_bgcolor = 'white',
            width = 700,
            height = height_px
        )
        if show:
            fi_fig.show()
        if chart_save:
            fi_fig.write_image(base_paths+'feature_importance_chart.png')
        return model, train_acc_list, val_acc_list, fig, fi_fig, metrics_score

        """
        def compute_metric(y_true, y_pred, y_prob=None):
            if optimizationMetric.lower() == 'accuracy':
                return accuracy_score(y_true, y_pred)
            elif optimizationMetric.lower() == 'precision':
                return precision_score(y_true, y_pred, average='weighted', zero_division=0)
            elif optimizationMetric.lower() == 'recall':
                return recall_score(y_true, y_pred, average='weighted', zero_division=0)
            elif optimizationMetric.lower() == 'auc':
                if num_classes == 2:
                    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                        y_prob = y_prob[:, 1]
                    return roc_auc_score(y_true, y_prob)
                else:
                    y_true_bin = label_binarize(y_true, classes=unique_labels)
                    return roc_auc_score(y_true_bin, y_prob, multi_class='ovr', average='weighted')

        train_metric_list = []
        val_metric_list = []
        algo = algorithm.lower()

        if algo == 'lightgbm':
            if optimizationMetric == 'auc':
                if num_classes == 2:
                    grid_scoring = 'roc_auc'
                else:
                    grid_scoring = 'roc_auc_ovr_weighted'
            else:
                grid_scoring = optimizationMetric
                
            model = lgb.LGBMClassifier(objective='multiclass' if num_classes > 2 else 'binary', num_class=num_classes if num_classes > 2 else None, n_estimators=n_estimators, random_state=random_state, verbose=-1)
            if tuning:
                if optimizationAlgorithm == 'scikit_learn' and learningRateScheduler:
                    param_grid = {'learning_rate': np.linspace(minimumLearningRate, maximumLearningRate, optimizationIterations)}
                    grid = GridSearchCV(model, param_grid, cv=5, scoring=grid_scoring, n_jobs=-1)
                    grid.fit(X_train, y_train)
                    best_lr = grid.best_params_['learning_rate']
                    model = lgb.LGBMClassifier(objective='multiclass' if num_classes > 2 else 'binary', num_class=num_classes if num_classes > 2 else None, n_estimators=n_estimators, random_state=random_state, verbose=-1, learning_rate=best_lr)
                elif optimizationAlgorithm == 'optuna' and learningRateScheduler:
                    def objective(trial):
                        lr = trial.suggest_float("learning_rate", minimumLearningRate, maximumLearningRate)
                        temp_model = lgb.LGBMClassifier(objective='multiclass' if num_classes > 2 else 'binary', num_class=num_classes if num_classes > 2 else None, n_estimators=n_estimators, random_state=random_state, learning_rate=lr, verbose=-1)
                        temp_model.fit(X_train, y_train)
                        y_val_pred = temp_model.predict(X_val)
                        y_val_prob = temp_model.predict_proba(X_val) if hasattr(temp_model, 'predict_proba') else y_val_pred
                        return compute_metric(y_val, y_val_pred, y_val_prob)                
                    study = optuna.create_study(direction='maximize')
                    study.optimize(objective, n_trials=optimizationIterations or 20)
                    best_lr = study.best_params['learning_rate']
                    model = lgb.LGBMClassifier(objective='multiclass' if num_classes > 2 else 'binary', num_class=num_classes if num_classes > 2 else None, n_estimators=n_estimators, random_state=random_state, verbose=-1, learning_rate=best_lr)

            if not tuning:
                optimizationMetric = 'accuracy'
            
            def lgb_eval(y_true, y_pred):
                if num_classes == 2:
                    y_pred_label = (y_pred > 0.5).astype(int)
                else:
                    y_pred_label = np.argmax(y_pred, axis=1)
                if optimizationMetric == 'auc':
                    return optimizationMetric, compute_metric(y_true, y_pred_label, y_pred), True
                else:
                    return optimizationMetric, compute_metric(y_true, y_pred_label), True

            def record_metric(env):
                for data_name, eval_name, result, _ in env.evaluation_result_list:
                    if eval_name == optimizationMetric:
                        if data_name == 'training':
                            train_metric_list.append(result)
                        elif data_name == 'valid_0':
                            val_metric_list.append(result)
            
            callbacks = [record_metric, lgb.log_evaluation(period=0)]
            if tuning and earlyStop > 0:
                callbacks.append(lgb.early_stopping(stopping_rounds=earlyStop))
            model.fit(X_train, y_train, eval_set=[(X_val, y_val), (X_train, y_train)], eval_metric=lgb_eval, callbacks=callbacks)

        elif algo == 'gradient_boosting':
            if optimizationMetric == 'auc':
                if num_classes == 2:
                    grid_scoring = 'roc_auc'
                else:
                    grid_scoring = 'roc_auc_ovr_weighted'
            else:
                grid_scoring = optimizationMetric
            
            model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state, loss='log_loss')
            if tuning:
                if optimizationAlgorithm == 'scikit_learn' and learningRateScheduler:
                    param_grid = {'learning_rate': np.linspace(minimumLearningRate, maximumLearningRate, optimizationIterations)}
                    grid = GridSearchCV(model, param_grid, cv=5, scoring=grid_scoring, n_jobs=-1)
                    grid.fit(X_train, y_train)
                    best_lr = grid.best_params_['learning_rate']
                    model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state, loss='log_loss', learning_rate=best_lr)
                elif optimizationAlgorithm == 'optuna' and learningRateScheduler:
                    def objective(trial):
                        lr = trial.suggest_float("learning_rate", minimumLearningRate, maximumLearningRate)
                        temp_model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state, loss='log_loss', learning_rate=lr)
                        temp_model.fit(X_train, y_train)
                        y_val_pred = temp_model.predict(X_val)
                        y_val_prob = temp_model.predict_proba(X_val) if hasattr(temp_model, 'predict_proba') else y_val_pred
                        return compute_metric(y_val, y_val_pred, y_val_prob)
                    study = optuna.create_study(direction='maximize')
                    study.optimize(objective, n_trials=optimizationIterations or 20)
                    best_lr = study.best_params['learning_rate']                    
                    model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state, loss='log_loss', learning_rate=best_lr)

            if not tuning:
                optimizationMetric = 'accuracy'            
            
            model.fit(X_train, y_train)
            
            best_val = -np.inf
            patience = earlyStop or n_estimators
            counter = 0

            staged_train_pred = model.staged_predict(X_train)
            staged_val_pred = model.staged_predict(X_val)
            staged_train_prob = model.staged_predict_proba(X_train)
            staged_val_prob = model.staged_predict_proba(X_val)

            for i, (train_pred, val_pred, train_prob, val_prob) in enumerate(zip(staged_train_pred, staged_val_pred, staged_train_prob, staged_val_prob)):
                train_val = compute_metric(y_train, train_pred, train_prob)
                val_val = compute_metric(y_val, val_pred, val_prob)
                train_metric_list.append(train_val)
                val_metric_list.append(val_val)
                if val_val > best_val:
                    best_val = val_val
                    counter = 0
                else:
                    counter += 1
                if tuning and earlyStop and counter >= patience:
                    print(f"Training until validation scores don't improve for {i} rounds")                    
                    break

        elif algo == 'ada_boost':
            if optimizationMetric == 'auc':
                if num_classes == 2:
                    grid_scoring = 'roc_auc'
                else:
                    grid_scoring = 'roc_auc_ovr_weighted'
            else:
                grid_scoring = optimizationMetric

            base = DecisionTreeClassifier(max_depth=1, random_state=random_state)
            model = AdaBoostClassifier(estimator=base, n_estimators=n_estimators, random_state=random_state, algorithm='SAMME')
            if tuning:
                if optimizationAlgorithm == 'scikit_learn' and learningRateScheduler:
                    min_lr = max(minimumLearningRate, 0.01)
                    param_grid = {'learning_rate': np.linspace(min_lr, maximumLearningRate, optimizationIterations)}
                    grid = GridSearchCV(model, param_grid, cv=5, scoring=grid_scoring, n_jobs=-1)
                    grid.fit(X_train, y_train)
                    best_lr = grid.best_params_['learning_rate']
                    model = AdaBoostClassifier(estimator=base, n_estimators=n_estimators, random_state=random_state, algorithm='SAMME', learning_rate=best_lr)
                elif optimizationAlgorithm == 'optuna' and learningRateScheduler:
                    def objective(trial):
                        lr = trial.suggest_float("learning_rate", minimumLearningRate, maximumLearningRate)
                        temp_model = AdaBoostClassifier(n_estimators=n_estimators, random_state=random_state, algorithm='SAMME', learning_rate=lr)
                        temp_model.fit(X_train, y_train)
                        y_val_pred = temp_model.predict(X_val)
                        y_val_prob = temp_model.predict_proba(X_val) if hasattr(temp_model, 'predict_proba') else y_val_pred
                        return compute_metric(y_val, y_val_pred, y_val_prob)
                    study = optuna.create_study(direction='maximize')
                    study.optimize(objective, n_trials=optimizationIterations or 20)
                    best_lr = study.best_params['learning_rate']                    
                    model = AdaBoostClassifier(estimator=base, n_estimators=n_estimators, random_state=random_state, algorithm='SAMME', learning_rate=best_lr)
                    
            if not tuning:
                optimizationMetric = 'accuracy'            
            model.fit(X_train, y_train)
            best_val = -np.inf
            patience = earlyStop or n_estimators
            counter = 0
            staged_train_pred = model.staged_predict(X_train)
            staged_val_pred = model.staged_predict(X_val)
            staged_train_prob = model.staged_predict_proba(X_train)
            staged_val_prob = model.staged_predict_proba(X_val)
            for i, (train_pred, val_pred, train_prob, val_prob) in enumerate(zip(staged_train_pred, staged_val_pred, staged_train_prob, staged_val_prob)):
                train_val = compute_metric(y_train, train_pred, train_prob)
                val_val = compute_metric(y_val, val_pred, val_prob)
                train_metric_list.append(train_val)
                val_metric_list.append(val_val)
                if val_val > best_val:
                    best_val = val_val
                    counter = 0
                else:
                    counter += 1
                if tuning and earlyStop and counter >= patience:
                    print(f"Training until validation scores don't improve for {i} rounds")                    
                    break                    

        y_train_pred = model.predict(X_train)
        if hasattr(model, "predict_proba"):
            y_train_prob = model.predict_proba(X_train)
            if num_classes == 2:
                y_train_prob = y_train_prob[:, 1]
        else:
            y_train_prob = y_train_pred

        metrics_score = {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
        }

        if num_classes == 2:
            metrics_score['auc'] = roc_auc_score(y_train, y_train_prob)
        else:
            y_train_bin = label_binarize(y_train, classes=unique_labels)
            metrics_score['auc'] = roc_auc_score(y_train_bin, y_train_prob, multi_class='ovr', average='weighted')

        title_text = algorithm.replace('_', ' ').title()
        iterations = list(range(1, len(train_metric_list) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=iterations, y=train_metric_list, mode='lines', name='Train'))
        fig.add_trace(go.Scatter(x=iterations, y=val_metric_list, mode='lines', name='Validation'))
        y_min = min(min(train_metric_list), min(val_metric_list)) - 0.02
        y_max = max(max(train_metric_list), max(val_metric_list)) + 0.02
        fig.update_layout(
            title=f"{title_text} Training & Validation Accuracy",
            xaxis_title="Iteration",
            yaxis_title="Accuracy",
            yaxis=dict(range=[y_min, y_max]),
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=700
        )
        if show:
            fig.show()
        if chart_save:
            fig.write_image(base_paths+'metirc_chart.png')
        fi_fig = None
        feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'Feature {i}' for i in range(X_train.shape[1])]
        importances = model.feature_importances_
        importances = importances / importances.sum()
        fi_fig = go.Figure()
        fi_fig.add_trace(go.Bar(x=importances, y=feature, orientation='h', marker_color='indianred'))
        height_px = max(400, 50 * len(feature_names))
        fi_fig.update_layout(
            title = f"{title_text} Feature Importance",
            xaxis_title = 'Contribution to Model Prediction',
            yaxis_title = 'Features',
            plot_bgcolor = 'white',
            paper_bgcolor = 'white',
            width = 700,
            height = height_px
        )
        if show:
            fi_fig.show()
        if chart_save:
            fi_fig.write_image(base_paths+'feature_importance_chart.png')
        return model, train_metric_list, val_metric_list, fig, fi_fig, metrics_score
        """

    def encode_file_to_base64(file_path):
        try:
            with open(file_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception:
            statusMessage = '데이터의 Base64 인코딩 과정에서 오류가 발생했습니다.'
            logger.error(f"{statusMessage}")
            send_status_to_bentoml(projectId, versionId, status='ERROR', statusMessage=statusMessage)
            raise

    """ Task """
    logger.info(f"projectId: {projectId}")
    logger.info(f"versionId: {versionId}")    
    send_status_to_bentoml(projectId, versionId, algorithm, status='READY', statusMessage='요청 값이 모두 들어왔습니다.', progress=0)
    send_status_to_bentoml(projectId, versionId, algorithm, status='RUNNING', statusMessage='모델 학습 중입니다.')
    if isinstance(advancedSettingForClassification, str):
        advancedSettingForClassification = json.loads(advancedSettingForClassification)
    if isinstance(feature, str):
        feature = json.loads(feature)

    model, train_acc, val_acc, fig, fi_fig, metrics_score = train_and_plot(algorithm=algorithm, n_estimators=50, show=False, chart_save=True)        

    """
    earlyStop = advancedSettingForClassification['earlyStop']
    optimizationAlgorithm = advancedSettingForClassification['optimizationAlgorithm']
    optimizationMetric = advancedSettingForClassification['optimizationMetric']
    optimizationIterations = advancedSettingForClassification['optimizationIterations']
    learningRateScheduler = advancedSettingForClassification['learningRateScheduler']
    minimumLearningRate = advancedSettingForClassification['minimumLearningRate']
    maximumLearningRate = advancedSettingForClassification['maximumLearningRate']

    model, train_acc, val_acc, fig, fi_fig, metrics_score = train_and_plot(
        algorithm=algorithm,
        tuning=tuning,
        optimizationMetric=optimizationMetric,
        optimizationAlgorithm=optimizationIterations, 
        learningRateScheduler=learningRateScheduler, 
        minimumLearningRate=minimumLearningRate, 
        maximumLearningRate=maximumLearningRate, 
        optimizationIterations=optimizationIterations, 
        earlyStop=earlyStop, 
        n_estimators=50, 
        show=True,
        random_state=42,
        chart_save=True
    )
    """

    model_save_dir = os.path.join(base_paths, 'model_weight')
    os.makedirs(model_save_dir, exist_ok=True)
    model_file_path = os.path.join(model_save_dir, f"{algorithm}_model.pkl")
    joblib.dump(model, model_file_path)
    minio_upload_path = {
        'metric_chart_path': base_paths+'metirc_chart.png',
        'feature_importance_chart_path': base_paths+'feature_importance_chart.png'
    }

    zip_file_path = os.path.join(model_save_dir, f"model_weight.zip")
    with open(model_file_path, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    with zipfile.ZipFile(zip_file_path, 'w') as zf:
        zf.write(model_file_path, arcname=f"{algorithm}_model.pkl")
    
    model_weight_path = f"{base_paths}/model_weight/model_weight.zip"
    object_name = f"{projectId}/{versionId}/train/model_weight/model_weight.zip"
    try:
        client.fput_object(MinioBucketUser, object_name, model_weight_path)
    except Exception:
        statusMessage = '학습 모델의 가중치를 저장하는 과정에서 오류가 발생하였습니다.'
        logger.error(f"{statusMessage}")
        send_status_to_bentoml(projectId, versionId, algorithm, status='ERROR', statusMessage=statusMessage)
        raise

    training_results = {}
    for key, local_path in minio_upload_path.items():
        filename = os.path.basename(local_path)
        base_index = local_path.find(f"{projectId}/{versionId}/training")
        object_name = local_path[base_index:]
        #client.fput_object(MinioBucketUser, object_name, local_path)
        object_name_b64 = base64.b64encode(object_name.encode('utf-8')).decode('utf-8')
        training_results[key] = object_name_b64
    accuracy_loss_chart_path = f"{base_paths}/metirc_chart.png"
    feature_importance_path = f"{base_paths}/feature_importance_chart.png"
    training_result = {
        'base64MetricChart': {
            'name': 'accuracy_loss_chart.png',
            'base64': encode_file_to_base64(f"{base_paths}/metirc_chart.png")
        },
        'base64FeatureImportanceChart': {
            'name': 'feature_importance.png',
            'base64': encode_file_to_base64(f"{base_paths}/feature_importance_chart.png")
        }
    }
    send_status_to_bentoml(projectId, versionId, algorithm, status='FINISH', trainingResult=training_result, statusMessage='모델 학습이 완료되었습니다.', accuracy=round(metrics_score['accuracy'], 5), precision=round(metrics_score['precision'], 5), recall=round(metrics_score['recall'], 5), auc=round(metrics_score['auc'], 5))

def pipelines():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--projectId', type=str)
    parser.add_argument('--versionId', type=str)
    parser.add_argument('--target', type=str)
    parser.add_argument('--feature', type=str)
    parser.add_argument('--algorithm', type=str)
    parser.add_argument('--tuning', type=str)
    parser.add_argument('--advancedSettingForClassification', type=str)
    parser.add_argument('--status', type=str)
    parser.add_argument('--statusMessage', type=str)
    parser.add_argument('--trainingResult', type=str)
    args = parser.parse_args()
    Training_task = Training(args.projectId, args.versionId, args.target, args.feature, args.algorithm, args.tuning, args.advancedSettingForClassification, args.status, args.statusMessage, args.trainingResult) \
        .apply(onprem.mount_pvc(f"{KubeflowVolumeName}", volume_name='data', volume_mount_path=f"{KubeflowVolumeMountPath}")) \
        .execution_options.caching_strategy.max_cache_staleness = 'P0D'

if __name__ == '__main__':
    pipeline_package_path = f"{KubeflowPieplineTraining}_pipelines.zip"
    kfp.compiler.Compiler().compile(pipelines, pipeline_package_path)
    HOST = f"http://{KubeflowHost}"
    USERNAME = KubeflowUsername
    PASSWORD = KubeflowPassword
    NAMESPACE = KubeflowNamespace
    session = requests.Session()
    response = session.get(HOST)
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {'login': USERNAME, 'password': PASSWORD}
    session.post(response.url, headers = headers, data = data)
    session_cookie = session.cookies.get_dict()['authservice_session']
    client = kfp.Client(host = f"{HOST}/pipeline", cookies = f"authservice_session={session_cookie}", namespace = NAMESPACE)
    experiment = client.create_experiment(name=f"{KubeflowPieplineTraining}")
    run = client.run_pipeline(experiment.id, f"{KubeflowPieplineTraining} pipelines", pipeline_package_path)
