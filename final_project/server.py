"""Flask web application for emotion detection."""

from flask import Flask, render_template, request

from final_project.EmotionDetection import emotion_detector

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static",
)


@app.get("/")
def home():
    """Render the emotion detection homepage."""
    return render_template("index.html")


@app.get("/emotionDetector")
def detect_emotion():
    """Analyze the submitted text and return its emotion scores."""
    text_to_analyze = request.args.get("textToAnalyze", "")
    emotions = emotion_detector(text_to_analyze)
    if emotions["dominant_emotion"] is None:
        return "Invalid text! Please try again!", 400

    return (
        "For the given statement, the system response is "
        f"'anger': {emotions['anger']}, "
        f"'disgust': {emotions['disgust']}, "
        f"'fear': {emotions['fear']}, "
        f"'joy': {emotions['joy']} and "
        f"'sadness': {emotions['sadness']}. "
        f"The dominant emotion is <strong>{emotions['dominant_emotion']}</strong>."
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
