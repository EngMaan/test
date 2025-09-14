import sys
import os
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QComboBox, QFormLayout
)
from PyQt5.QtCore import Qt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

import shap
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(6, 4), dpi=100)
        super().__init__(self.fig)


class MotorPredictGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Motor Predictive Maintenance — XGBoost + SHAP")
        self.resize(980, 720)

        # Data / Model holders
        self.df = None
        self.target_col = None
        self.feature_cols = []
        self.label_encoder = None
        self.model = None
        self.explainer = None

        self.last_pred_sample = None   # pandas Series of the last sample predicted
        self.last_pred_index = None    # predicted class index (int)

        # UI
        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Row: Buttons
        btn_row = QHBoxLayout()
        self.btn_load = QPushButton("Load CSV")
        self.btn_load.clicked.connect(self.on_load_csv)
        self.btn_train = QPushButton("Train Model")
        self.btn_train.clicked.connect(self.on_train_model)
        self.btn_train.setEnabled(False)

        self.btn_predict = QPushButton("Predict")
        self.btn_predict.clicked.connect(self.on_predict)
        self.btn_predict.setEnabled(False)

        self.btn_explain = QPushButton("Explain Prediction (SHAP)")
        self.btn_explain.clicked.connect(self.on_explain)
        self.btn_explain.setEnabled(False)

        btn_row.addWidget(self.btn_load)
        btn_row.addWidget(self.btn_train)
        btn_row.addWidget(self.btn_predict)
        btn_row.addWidget(self.btn_explain)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        # Target selector row
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Target column:"))
        self.target_combo = QComboBox()
        self.target_combo.setEditable(False)
        self.target_combo.setEnabled(False)
        target_row.addWidget(self.target_combo)
        target_row.addStretch(1)
        root.addLayout(target_row)

        # Dynamic form for features
        self.form = QFormLayout()
        self.feature_inputs_container = QWidget()
        self.feature_inputs_container.setLayout(self.form)
        root.addWidget(self.feature_inputs_container)

        # Output labels
        out_row = QHBoxLayout()
        self.lbl_status = QLabel("Status: Load a CSV to begin.")
        self.lbl_prediction = QLabel("Prediction: —")
        self.lbl_status.setStyleSheet("font-weight: bold;")
        self.lbl_prediction.setStyleSheet("font-size: 16px; font-weight: bold;")
        out_row.addWidget(self.lbl_status, stretch=2)
        out_row.addWidget(self.lbl_prediction, alignment=Qt.AlignRight)
        root.addLayout(out_row)

        # Matplotlib canvas for SHAP plots
        self.canvas = MplCanvas()
        root.addWidget(self.canvas, stretch=1)

    # -------------- Actions --------------
    def on_load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV files (*.csv)")
        if not path:
            return
        try:
            self.df = pd.read_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read CSV:\n{e}")
            return

        if self.df.empty or self.df.shape[1] < 2:
            QMessageBox.warning(self, "Invalid", "CSV must have at least 2 columns (features + target).")
            return

        # Populate target combo
        self.target_combo.clear()
        self.target_combo.addItems(list(self.df.columns))
        # Try to preselect common target names
        for guess in ["Label", "Target", "label", "target", "y"]:
            if guess in self.df.columns:
                idx = self.target_combo.findText(guess)
                if idx >= 0:
                    self.target_combo.setCurrentIndex(idx)
                    break
        self.target_combo.setEnabled(True)

        # Build feature inputs initially assuming current target selection
        self.build_feature_inputs()
        self.btn_train.setEnabled(True)
        self.lbl_status.setText(f"Loaded: {os.path.basename(path)} ({self.df.shape[0]} rows, {self.df.shape[1]} cols)")

    def build_feature_inputs(self):
        # Clear old inputs
        while self.form.rowCount() > 0:
            self.form.removeRow(0)

        self.target_col = self.target_combo.currentText()
        self.feature_cols = [c for c in self.df.columns if c != self.target_col]

        self.inputs = {}  # name -> QLineEdit
        for col in self.feature_cols:
            edit = QLineEdit()
            edit.setPlaceholderText("Enter numeric value")
            self.form.addRow(QLabel(col), edit)
            self.inputs[col] = edit

        self.target_combo.currentTextChanged.connect(self._on_target_changed)

    def _on_target_changed(self, _):
        if self.df is not None:
            self.build_feature_inputs()

    def on_train_model(self):
        if self.df is None:
            QMessageBox.information(self, "Info", "Load a CSV first.")
            return

        self.target_col = self.target_combo.currentText()
        if self.target_col not in self.df.columns:
            QMessageBox.warning(self, "Invalid", "Select a valid target column.")
            return

        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        # Label encode target (categorical -> integers)
        self.label_encoder = LabelEncoder()
        try:
            y_enc = self.label_encoder.fit_transform(y)
        except Exception as e:
            QMessageBox.critical(self, "Encoding Error", f"Failed to encode target:\n{e}")
            return

        # Train / test split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_enc, test_size=0.2, random_state=42, stratify=y_enc if len(np.unique(y_enc)) > 1 else None
            )
        except Exception as e:
            QMessageBox.critical(self, "Split Error", f"Failed to split data:\n{e}")
            return

        # Train model
        num_classes = len(np.unique(y_enc))
        self.model = XGBClassifier(
            n_estimators=30,
            max_depth=5,
            learning_rate=0.5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="multi:softprob" if num_classes > 2 else "binary:logistic",
            eval_metric="mlogloss" if num_classes > 2 else "logloss",
            tree_method="hist"
        )

        try:
            self.model.fit(X_train, y_train)
        except Exception as e:
            QMessageBox.critical(self, "Training Error", f"Failed to train XGBoost:\n{e}")
            return

        # Evaluate
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred,
            target_names=[str(c) for c in self.label_encoder.classes_],
            zero_division=0
        )

        self.lbl_status.setText(f"Trained. Test Accuracy: {acc:.3f}")
        QMessageBox.information(self, "Model Trained",
                                f"Classes: {list(self.label_encoder.classes_)}\n\nAccuracy: {acc:.3f}\n\n{report}")

        # Prepare SHAP explainer once per model
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception as e:
            self.explainer = None
            QMessageBox.warning(self, "SHAP Warning", f"TreeExplainer init failed:\n{e}")

        self.btn_predict.setEnabled(True)
        # self.btn_explain.setEnabled(False)  # enabled after a prediction
        self.canvas.fig.clf()
        self.canvas.draw()
        self.lbl_prediction.setText("Prediction: —")

    def _collect_feature_input(self):
        if not self.inputs:
            QMessageBox.information(self, "Info", "Load data and train first.")
            return None

        vals = {}
        for col, widget in self.inputs.items():
            txt = widget.text().strip()
            if txt == "":
                QMessageBox.warning(self, "Missing", f"Please enter a value for '{col}'.")
                return None
            try:
                vals[col] = float(txt)
            except ValueError:
                QMessageBox.warning(self, "Invalid", f"'{col}' must be numeric.")
                return None
        return pd.DataFrame([vals])[self.feature_cols]  # ensure correct column order

    def on_predict(self):
        if self.model is None or self.label_encoder is None:
            QMessageBox.information(self, "Info", "Train the model first.")
            return

        X_new = self._collect_feature_input()
        if X_new is None:
            return

        try:
            pred_idx = int(self.model.predict(X_new)[0])
            pred_label = self.label_encoder.inverse_transform([pred_idx])[0]
        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", f"Failed to predict:\n{e}")
            return

        self.lbl_prediction.setText(f"Prediction: {pred_label}")
        self.lbl_status.setText("Predicted successfully.")
        self.last_pred_sample = X_new.iloc[0]
        self.last_pred_index = pred_idx
        self.btn_explain.setEnabled(self.explainer is not None)

    def on_explain(self):
        if self.explainer is None:
            QMessageBox.information(self, "Info", "SHAP explainer not available. Train again.")
            return
        if self.last_pred_sample is None:
            QMessageBox.information(self, "Info", "Make a prediction first.")
            return

        try:
            # Compute SHAP for this single instance
            X_one = pd.DataFrame([self.last_pred_sample.values], columns=self.feature_cols)
            shap_values = self.explainer.shap_values(X_one)

            # Handle multi-class / single-array differences across SHAP versions
            if isinstance(shap_values, list):
                # Multi-class SHAP -> pick explanation for predicted class
                class_idx = int(self.last_pred_index) if self.last_pred_index is not None else 0
                values = shap_values[class_idx][0]
                base = float(self.explainer.expected_value[class_idx])
            else:
                # Single array SHAP (new versions)
                values = shap_values[0]
                if isinstance(self.explainer.expected_value, (list, np.ndarray)):
                    base = float(self.explainer.expected_value[0])
                else:
                    base = float(self.explainer.expected_value)


            # Draw SHAP waterfall (fallback to bar if waterfall fails)
            self.canvas.fig.clf()
            ax = self.canvas.fig.add_subplot(111)

            try:
                # Build Explanation object for stable plotting
                exp = shap.Explanation(
                    values=np.array(values, dtype=float),
                    base_values=float(base),
                    data=self.last_pred_sample.values.astype(float),
                    feature_names=self.feature_cols
                )
                # Use waterfall if available; it renders into matplotlib axes implicitly
                shap.plots.waterfall(exp, show=False)
                ax.set_title("SHAP Waterfall — single prediction")
            except Exception:
                ax.clear()
                # Fallback: simple bar plot
                order = np.argsort(np.abs(values))[::-1]
                ax.bar(range(len(values)), np.array(values)[order])
                ax.set_xticks(range(len(values)))
                ax.set_xticklabels(np.array(self.feature_cols)[order], rotation=45, ha="right")
                ax.set_ylabel("SHAP value")
                ax.set_title("SHAP Feature Contributions (bar)")

            self.canvas.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "SHAP Error", f"Failed to compute/plot SHAP:\n{e}")


def main():
    app = QApplication(sys.argv)
    win = MotorPredictGUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
