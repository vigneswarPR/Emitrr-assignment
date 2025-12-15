

from transformers import pipeline



NEG_TRIGGERS = [
    "worried", "worry", "anxious", "anxiety", "concerned", "scared", "afraid",
    "pain", "hurt", "discomfort", "stiffness", "ache", "sore",
]
POS_TRIGGERS = [
    "relieved", "relief", "better", "improving", "improved",
    "fine now", "okay now", "that's a relief", "great to hear",
]


class SentimentAnalyzer:
    def __init__(self, model: str = "brettclaus/Hospital_Reviews"):
        print("Initializing medical sentiment pipeline...")
        
        self.pipe = pipeline("sentiment-analysis", model=model)

    def analyze_sentiment(self, text: str) -> str:
        """
        Map model sentiment + clinical keywords to: Anxious / Neutral / Reassured.
        """
        out = self.pipe(text)[0]  
        base = out["label"].upper()
        t = text.lower()

        
        if any(w in t for w in NEG_TRIGGERS):
            return "Anxious"
        if any(w in t for w in POS_TRIGGERS):
            return "Reassured"

        
        if base.startswith("NEG"):
            return "Anxious"
        if base.startswith("POS"):
            return "Reassured"
        return "Neutral"

    def detect_intent(self, text: str) -> str:
        """
        Simple rule-based intent: Greeting / Seeking reassurance / Reporting symptoms / Expressing concern / Answering question.
        """
        t = text.lower().strip()

        
        if any(g in t for g in ["good morning", "good afternoon", "good evening", "hello", "hi "]):
            return "Greeting"

        
        if any(phrase in t for phrase in ["thank you", "thanks", "appreciate it"]):
            return "Expressing concern"


        if "?" in t and any(w in t for w in ["worry", "worried", "concerned", "ok", "okay", "future", "problem"]):
            return "Seeking reassurance"

        
        if "?" not in t and any(t.startswith(w) for w in ["yes", "no", "yeah", "nope"]):
            return "Answering question"

        
        if any(w in t for w in ["pain", "hurt", "discomfort", "ache", "stiffness", "injury", "whiplash", "back", "neck", "headache"]):
            return "Reporting symptoms"

        
        return "Reporting symptoms"


if __name__ == "__main__":
    sa = SentimentAnalyzer()
    s = "I'm doing better, but I still have some discomfort now and then."
    print("Sentiment:", sa.analyze_sentiment(s))
    print("Intent:", sa.detect_intent(s))

