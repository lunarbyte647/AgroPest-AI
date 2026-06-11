from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/classify/agropest_final/weights/best.pt")


def predict_crop(image_path):
    """
    Predict crop disease from image.
    Returns a dictionary containing prediction details.
    """

    results = model(image_path)

    # Top prediction
    probs = results[0].probs

    class_id = probs.top1
    confidence = float(probs.top1conf)

    prediction = model.names[class_id]

    # Crop health calculations
    green = round(confidence * 100, 2)
    damage = round(100 - green, 2)

    score = green

    if score >= 80:
        status = "Excellent"
        severity = "Low"
        risk = "Low"
        recovery = "Excellent"

    elif score >= 60:
        status = "Good"
        severity = "Moderate"
        risk = "Moderate"
        recovery = "Good"

    else:
        status = "Poor"
        severity = "High"
        risk = "High"
        recovery = "Needs Attention"

    recommendation = (
        "Continue regular monitoring."
        if "healthy" in prediction.lower()
        else "Consult agricultural experts and apply appropriate treatment."
    )

    return {
        "prediction": prediction,
        "confidence": round(confidence * 100, 2),
        "green": green,
        "damage": damage,
        "score": round(score, 2),
        "status": status,
        "severity": severity,
        "risk": risk,
        "recovery": recovery,
        "recommendation": recommendation
    }