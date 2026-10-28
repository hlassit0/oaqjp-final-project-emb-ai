import json
import unittest
from unittest.mock import Mock, patch

from final_project.EmotionDetection import emotion_detector


class EmotionDetectorTests(unittest.TestCase):
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
