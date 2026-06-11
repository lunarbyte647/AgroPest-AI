from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("runs/classify/train/weights/best.pt")


def analyze_health(image_path):
    """
    Calculates green and damaged leaf percentages.
    """

    image = cv2.imread(image_path)

    if image is None:
        return 0, 0

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Green leaf mask
    lower_green = (35, 40, 40)
    upper_green = (90, 255, 255)

    green_mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    total_pixels = image.shape[0] * image.shape[1]

    green_pixels = cv2.countNonZero(
        green_mask
    )

    green_percentage = (
        green_pixels / total_pixels
    ) * 100

    damaged_percentage = (
        100 - green_percentage
    )

    return (
        round(green_percentage, 2),
        round(damaged_percentage, 2)
    )


def calculate_health(green):
    """
    Crop Health Index
    """
    return round(green, 2)


def health_status(score):

    if score >= 80:
        return "Excellent"

    elif score >= 60:
        return "Good"

    elif score >= 40:
        return "Moderate"

    else:
        return "Poor"


def severity_level(score):

    if score >= 80:
        return "Low"

    elif score >= 60:
        return "Moderate"

    else:
        return "High"


def pest_risk(confidence):

    confidence = confidence * 100

    if confidence >= 90:
        return "Very High"

    elif confidence >= 70:
        return "High"

    elif confidence >= 50:
        return "Moderate"

    else:
        return "Low"


def recovery_potential(score):

    if score >= 80:
        return "Excellent"

    elif score >= 60:
        return "Good"

    elif score >= 40:
        return "Fair"

    else:
        return "Poor"


def get_recommendation(prediction):

    recommendations = {

        "Cashew healthy":
        "Maintain regular monitoring and irrigation.",

        "Cashew anthracnose":
        "Apply recommended fungicides and remove infected leaves.",

        "Cashew leaf miner":
        "Use biological pest control methods.",

        "Cassava bacterial blight":
        "Remove infected plants and improve field sanitation.",

        "Cassava healthy":
        "Continue standard crop maintenance practices."

    }

    return recommendations.get(
        prediction,
        "Consult an agricultural expert for treatment recommendations."
    )


def predict_crop(image_path):

    results = model.predict(
        image_path,
        verbose=False
    )

    green, damage = analyze_health(
        image_path
    )

    score = calculate_health(
        green
    )

    status = health_status(
        score
    )

    severity = severity_level(
        score
    )

    for r in results:

        class_id = r.probs.top1

        confidence = float(
            r.probs.top1conf
        )

        prediction = r.names[
            class_id
        ]

        risk = pest_risk(
            confidence
        )

        recovery = recovery_potential(
            score
        )

        recommendation = get_recommendation(
            prediction
        )

        return {

            "prediction":
            prediction,

            "confidence":
            round(confidence * 100, 2),

            "green":
            green,

            "damage":
            damage,

            "score":
            score,

            "status":
            status,

            "severity":
            severity,

            "risk":
            risk,

            "recovery":
            recovery,

            "recommendation":
            recommendation

        }