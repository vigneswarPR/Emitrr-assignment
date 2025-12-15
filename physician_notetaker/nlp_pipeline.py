import spacy
import re
from collections import Counter

class MedicalNLPPipeline:
    def __init__(self):
        """
        Initialize the medical NLP pipeline with scispacy models.
        """
        print("Loading medical NLP model")
        try:
            
            self.nlp = spacy.load("en_ner_bc5cdr_md")
            print(" Loaded en_ner_bc5cdr_md model")
        except OSError:
            print(" Model not found. Please install with:")
            print("pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz")
            raise
    
    def extract_entities(self, transcript):
        """
        Extract medical entities from the transcript using NER.
        Returns a structured dictionary with all medical information.
        """
        doc = self.nlp(transcript)
        
        
        diseases = []
        chemicals = []
        
        for ent in doc.ents:
            if ent.label_ == "DISEASE":
                diseases.append(ent.text)
            elif ent.label_ == "CHEMICAL":
                chemicals.append(ent.text)
        
        
        symptoms = self._extract_symptoms(transcript)
        treatment = self._extract_treatment(transcript)
        diagnosis = self._extract_diagnosis(transcript, diseases)
        prognosis = self._extract_prognosis(transcript)
        patient_name = self._extract_patient_name(transcript)
        current_status = self._extract_current_status(transcript)
        
        return {
            "Patient_Name": patient_name,
            "Symptoms": symptoms,
            "Diagnosis": diagnosis,
            "Treatment": treatment,
            "Current_Status": current_status,
            "Prognosis": prognosis,
            "Detected_Diseases": list(set(diseases)),
            "Detected_Chemicals": list(set(chemicals))
        }
    
    def _extract_patient_name(self, transcript):
        """Extract patient name from the transcript."""
        
        name_pattern = r'\b(Ms\.|Mr\.|Mrs\.|Dr\.)\s+([A-Z][a-z]+)'
        match = re.search(name_pattern, transcript)
        if match:
            return f"{match.group(1)} {match.group(2)}"
        return "Unknown"
    
    def _extract_symptoms(self, transcript):
        """Extract symptoms mentioned by the patient."""
        symptoms = []
        
        
        symptom_keywords = {
            "pain": ["neck pain", "back pain", "head pain", "headache"],
            "discomfort": ["discomfort", "stiffness"],
            "trouble": ["trouble sleeping"],
            "backache": ["backache", "backaches"]
        }
        
        transcript_lower = transcript.lower()
        
        
        if "neck" in transcript_lower and "pain" in transcript_lower:
            symptoms.append("Neck pain")
        if "back" in transcript_lower and "pain" in transcript_lower:
            symptoms.append("Back pain")
        if "head" in transcript_lower and ("impact" in transcript_lower or "hit" in transcript_lower):
            symptoms.append("Head impact")
        if "discomfort" in transcript_lower:
            symptoms.append("Discomfort")
        if "stiffness" in transcript_lower:
            symptoms.append("Stiffness")
        if "trouble sleeping" in transcript_lower:
            symptoms.append("Trouble sleeping")
        if "backache" in transcript_lower:
            symptoms.append("Occasional backache")
        
        return list(set(symptoms))
    
    def _extract_diagnosis(self, transcript, detected_diseases):
        """Extract diagnosis from the transcript."""
        diagnosis = []
        
        
        if "whiplash" in transcript.lower():
            diagnosis.append("Whiplash injury")
        
        
        diagnosis.extend(detected_diseases)
        
        return list(set(diagnosis)) if diagnosis else ["Under investigation"]
    
    def _extract_treatment(self, transcript):
        """Extract treatment information."""
        treatment = []
        
        
        if "physiotherapy" in transcript.lower():
            
            session_match = re.search(r'(\w+)\s+sessions?\s+of\s+physiotherapy', transcript.lower())
            if session_match:
                treatment.append(f"{session_match.group(1).capitalize()} physiotherapy sessions")
            else:
                treatment.append("Physiotherapy sessions")
        
        if "painkiller" in transcript.lower():
            treatment.append("Painkillers")
        
        if "x-ray" in transcript.lower() or "xray" in transcript.lower():
            if "didn't do" in transcript.lower() or "no x-ray" in transcript.lower():
                treatment.append("No X-rays performed")
            else:
                treatment.append("X-ray examination")
        
        return treatment
    
    def _extract_prognosis(self, transcript):
        """Extract prognosis information."""
        prognosis = []
        
        
        if "full recovery" in transcript.lower():
            
            timeframe_match = re.search(r'within\s+(\w+\s+\w+)', transcript.lower())
            if timeframe_match:
                prognosis.append(f"Full recovery expected within {timeframe_match.group(1)}")
            else:
                prognosis.append("Full recovery expected")
        
        if "no long-term" in transcript.lower() or "no signs of long-term" in transcript.lower():
            prognosis.append("No long-term damage expected")
        
        return prognosis
    
    def _extract_current_status(self, transcript):
        """Extract current patient status."""
        
        if "occasional" in transcript.lower() and "backache" in transcript.lower():
            return "Occasional backache"
        elif "doing better" in transcript.lower():
            return "Improving"
        elif "recovered" in transcript.lower():
            return "Recovered"
        return "Stable"
    
    def extract_keywords(self, transcript):
        """
        Extract important medical keywords and phrases.
        """
        doc = self.nlp(transcript)
        keywords = []
        
        
        for ent in doc.ents:
            keywords.append(ent.text)
        
        
        for chunk in doc.noun_chunks:
            
            chunk_text = chunk.text.lower()
            if any(word in chunk_text for word in [
                'pain', 'injury', 'treatment', 'therapy', 'accident', 
                'recovery', 'damage', 'examination', 'symptom', 'condition'
            ]):
                keywords.append(chunk.text)
        
        
        keyword_counts = Counter(keywords)
        return [kw for kw, count in keyword_counts.most_common(15)]
    
    def summarize_transcript(self, transcript):
        """
        Generate a concise summary of the medical transcript.
        For now, uses extractive summarization based on key sentences.
        """
        
        doc = self.nlp(transcript)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        
        important_keywords = [
            'diagnosis', 'treatment', 'symptoms', 'pain', 'injury',
            'recovery', 'prognosis', 'examination', 'accident'
        ]
        
        scored_sentences = []
        for sent in sentences:
            sent_lower = sent.lower()
            score = sum(1 for keyword in important_keywords if keyword in sent_lower)
            if score > 0 and ('Patient:' in sent or 'Physician:' in sent):
                scored_sentences.append((score, sent))
        
        
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        summary_sentences = [sent for score, sent in scored_sentences[:5]]
        
        return " ".join(summary_sentences)
    
    def generate_structured_summary(self, transcript):
        """
        Generate a complete structured medical summary.
        This is the main method that combines all extraction methods.
        """
        entities = self.extract_entities(transcript)
        keywords = self.extract_keywords(transcript)
        summary = self.summarize_transcript(transcript)
        
        return {
            **entities,
            "Keywords": keywords,
            "Summary": summary
        }


if __name__ == "__main__":
    
    pipeline = MedicalNLPPipeline()
    
    
    test_text = """
    Patient: I had a car accident. My neck and back hurt a lot for four weeks.
    Doctor: Did you receive treatment?
    Patient: Yes, I had ten physiotherapy sessions, and now I only have occasional back pain.
    """
    
    result = pipeline.generate_structured_summary(test_text)
    print("\nTest Results")
    for key, value in result.items():
        print(f"{key}: {value}")
