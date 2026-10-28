# Emotion Detection Final Project Submission

## Task 1: Repository

- Public repository: <https://github.com/hlassit0/oaqjp-final-project-emb-ai>
- README: <https://github.com/hlassit0/oaqjp-final-project-emb-ai/blob/main/README.md>

## Task 2: Watson NLP emotion detection

### Activity 1: `final_project/emotion_detection.py`

This is the final version of the requested function, including the HTTP 400 handling added in Task 7.

```python
import json

import requests


def emotion_detector(text_to_analyze):
    url = (
        "https://sn-watson-emotion.labs.skills.network/"
        "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    )
    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
    }
    payload = {"raw_document": {"text": text_to_analyze}}

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 400:
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }

    response_data = json.loads(response.text)
    emotion_scores = response_data["emotionPredictions"][0]["emotion"]
    emotions = {
        "anger": emotion_scores["anger"],
        "disgust": emotion_scores["disgust"],
        "fear": emotion_scores["fear"],
        "joy": emotion_scores["joy"],
        "sadness": emotion_scores["sadness"],
    }
    dominant_emotion = max(emotions, key=emotions.get)
    return {
        "anger": emotions["anger"],
        "disgust": emotions["disgust"],
        "fear": emotions["fear"],
        "joy": emotions["joy"],
        "sadness": emotions["sadness"],
        "dominant_emotion": dominant_emotion,
    }
```

### Activity 2: Import and test evidence

The module imported successfully and its request arguments and returned response text were verified with a mocked HTTP response:

```text
Import and mocked request/response check passed.
```

A live request with “I love this new technology.” was attempted, but the Watson endpoint timed out in this environment. Therefore, a live service prediction could not be verified.

## Task 3: Formatted output

### Activity 1: Formatted function

The final `emotion_detector` source above parses the JSON response, returns all five emotion scores, and sets `dominant_emotion` to the emotion with the highest score.

### Activity 2: Output test evidence

Using a mocked Watson response for “I am so happy I am doing this”:

```text
{'anger': 0.01, 'disgust': 0.02, 'fear': 0.03, 'joy': 0.91, 'sadness': 0.03, 'dominant_emotion': 'joy'}
Dominant emotion is joy.
```

This verifies the output shape and dominant-emotion calculation. The Watson service response itself was mocked because the live endpoint timed out.

## Task 4: Package validation

### Activity 1: Public package initializer

<https://github.com/hlassit0/oaqjp-final-project-emb-ai/blob/main/final_project/EmotionDetection/__init__.py>

The initializer exports the application function:

```python
from final_project.emotion_detection import emotion_detector

__all__ = ["emotion_detector"]
```

### Activity 2: Package import and test evidence

The package imported, and a mocked response for “I hate working long hours” returned anger as the dominant emotion:

```text
{'anger': 0.84, 'disgust': 0.01, 'fear': 0.02, 'joy': 0.03, 'sadness': 0.1, 'dominant_emotion': 'anger'}
EmotionDetection package import passed; dominant emotion is anger.
```

## Task 5: Unit tests

### Activity 1: `final_project/test_emotion_detection.py`

```python
import json
import unittest
from unittest.mock import Mock, patch

from final_project.EmotionDetection import emotion_detector


class EmotionDetectorTests(unittest.TestCase):
    def test_bad_request_returns_none_for_every_emotion(self):
        with patch(
            "final_project.emotion_detection.requests.post",
            return_value=Mock(status_code=400, text=""),
        ) as post:
            result = emotion_detector("")

        self.assertEqual(
            result,
            {
                "anger": None,
                "disgust": None,
                "fear": None,
                "joy": None,
                "sadness": None,
                "dominant_emotion": None,
            },
        )
        post.assert_called_once()

    def test_dominant_emotion_for_each_statement(self):
        test_cases = (
            ("I am glad this happened", "joy"),
            ("I am really mad about this", "anger"),
            ("I feel disgusted just hearing about this", "disgust"),
            ("I am so sad about this", "sadness"),
            ("I am really afraid that this will happen", "fear"),
        )

        for statement, dominant_emotion in test_cases:
            with self.subTest(statement=statement):
                emotion_scores = {
                    "anger": 0.05,
                    "disgust": 0.05,
                    "fear": 0.05,
                    "joy": 0.05,
                    "sadness": 0.05,
                }
                emotion_scores[dominant_emotion] = 0.8
                response_body = {
                    "emotionPredictions": [{"emotion": emotion_scores}]
                }

                with patch(
                    "final_project.emotion_detection.requests.post",
                    return_value=Mock(text=json.dumps(response_body)),
                ) as post:
                    result = emotion_detector(statement)

                self.assertEqual(result["dominant_emotion"], dominant_emotion)
                self.assertEqual(result["anger"], emotion_scores["anger"])
                self.assertEqual(result["disgust"], emotion_scores["disgust"])
                self.assertEqual(result["fear"], emotion_scores["fear"])
                self.assertEqual(result["joy"], emotion_scores["joy"])
                self.assertEqual(result["sadness"], emotion_scores["sadness"])
                post.assert_called_once_with(
                    "https://sn-watson-emotion.labs.skills.network/"
                    "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict",
                    headers={
                        "grpc-metadata-mm-model-id":
                            "emotion_aggregated-workflow_lang_en_stock"
                    },
                    json={"raw_document": {"text": statement}},
                )


if __name__ == "__main__":
    unittest.main()
```

### Activity 2: Unit test output

Command: `python3 -B -m unittest -v final_project.test_emotion_detection`

```text
test_bad_request_returns_none_for_every_emotion (final_project.test_emotion_detection.EmotionDetectorTests.test_bad_request_returns_none_for_every_emotion) ... ok
test_dominant_emotion_for_each_statement (final_project.test_emotion_detection.EmotionDetectorTests.test_dominant_emotion_for_each_statement) ... ok

----------------------------------------------------------------------
Ran 2 tests in 0.001s

OK
```

The second test contains subtests for all five required statements and dominant emotions.

## Task 6: Flask deployment

### Activity 1: `final_project/server.py`

This final server includes the Task 7 blank-input handling and the docstrings used for the Task 8 Pylint check.

```python
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
```

### Activity 2: Deployment test evidence

The Flask homepage was verified at `http://localhost:5000`:

```text
Homepage: HTTP 200
Emotion detector: HTTP 200
For the given statement, the system response is 'anger': 0.006274985, 'disgust': 0.0025598293, 'fear': 0.009251528, 'joy': 0.9680386 and 'sadness': 0.049744144. The dominant emotion is <strong>joy</strong>.
```

The emotion result above used a mocked detector response. The screenshot [6b_deployment_test.png](./6b_deployment_test.png) also shows a simulated Watson response because a live Watson-backed result could not be verified in this environment.

## Task 7: Error handling

### Activity 1: HTTP 400 behavior in `emotion_detector`

The final function under Task 2 includes the status check. For `response.status_code == 400`, it returns `anger`, `disgust`, `fear`, `joy`, `sadness`, and `dominant_emotion`, each with the value `None`.

### Activity 2: Flask blank-input handling

The final `server.py` under Task 6 checks `emotions["dominant_emotion"] is None` and returns:

```python
return "Invalid text! Please try again!", 400
```

The client script was also updated to display both HTTP 200 and HTTP 400 response bodies.

### Activity 3: Blank-input test evidence

With the upstream HTTP 400 simulated, the running Flask endpoint returned:

```text
HTTP/1.1 400 BAD REQUEST

Invalid text! Please try again!
```

The browser UI displayed the same message for blank input. See [7c_error_handling_interface.png](./7c_error_handling_interface.png). The browser response was simulated as HTTP 400 to exercise the UI handling.

## Task 8: Static code analysis

### Activity 1: Modified server

The final `server.py` under Task 6 contains module and function docstrings.

### Activity 2: Pylint output

Command: `python3 -m pylint --source-roots=. final_project/server.py`

```text
------------------------------------
Your code has been rated at 10.00/10
```
