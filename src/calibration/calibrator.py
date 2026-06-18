from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV


def calibrate_classifier(model, X_calibration, y_calibration, method: str = "sigmoid"):
    try:
        calibrated = CalibratedClassifierCV(estimator=model, method=method, cv="prefit")
    except TypeError:  # pragma: no cover
        calibrated = CalibratedClassifierCV(base_estimator=model, method=method, cv="prefit")
    calibrated.fit(X_calibration, y_calibration)
    return calibrated
