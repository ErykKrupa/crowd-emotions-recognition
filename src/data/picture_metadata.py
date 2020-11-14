from data.emotion import Emotion


class PictureMetadata:
    def __init__(self, name: str, emotion: Emotion):
        self.name = name
        self.emotion = emotion
