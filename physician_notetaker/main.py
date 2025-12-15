import json
import os
from nlp_pipeline import MedicalNLPPipeline
from sentiment import SentimentAnalyzer
from soap_generator import SOAPGenerator


def load_transcript(filepath):
    """Load the transcript from a text file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()



def analyze_patient_statements(transcript, sentiment_analyzer):
    """
    Extract and analyze sentiment for each patient statement.
    """
    lines = transcript.split('\n')
    patient_analyses = []
    
    for line in lines:
        if line.strip().startswith('Patient:'):
            patient_text = line.replace('Patient:', '').strip()
            if patient_text:
                sentiment = sentiment_analyzer.analyze_sentiment(patient_text)
                intent = sentiment_analyzer.detect_intent(patient_text)
                
                patient_analyses.append({
                    "Statement": patient_text,
                    "Sentiment": sentiment,
                    "Intent": intent
                })
    
    return patient_analyses


def main():
    
    print("PHYSICIAN NOTETAKER - MEDICAL NLP PIPELINE\n")
    
    
    print("\n[1/4] Loading transcript...")
    transcript_path = os.path.join(os.path.dirname(__file__), 'sample_transcript.txt')
    transcript = load_transcript(transcript_path)
    print(f"✓ Loaded transcript ({len(transcript)} characters)")
    
    
    print("\n[2/4] Initializing Medical NLP Pipeline...")
    try:
        nlp_pipeline = MedicalNLPPipeline()
        print("✓ NLP Pipeline ready")
    except Exception as e:
        print(f"✗ Error initializing NLP pipeline: {e}")
        print("\nPlease install the required model:")
        print("pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz")
        return
    
    
    print("\n[3/4] Extracting medical entities and generating summary...")
    medical_summary = nlp_pipeline.generate_structured_summary(transcript)
    print("✓ Medical summary generated")
    
    
    print("\n[4/4] Analyzing patient sentiment and intent...")
    
    try:
        sentiment_analyzer = SentimentAnalyzer()
        patient_sentiments = analyze_patient_statements(transcript, sentiment_analyzer)
        print(f" Analyzed {len(patient_sentiments)} patient statements")
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        patient_sentiments = []
    
    
    print("\n[5/5] Generating SOAP Note...")
    soap_generator = SOAPGenerator(nlp_pipeline)
    soap_note = soap_generator.generate_soap_note(transcript, medical_summary)
    print("✓ SOAP note generated")
    
        
    final_output = {
        "Medical_NLP_Summary": medical_summary,
        "Patient_Sentiment_Analysis": patient_sentiments,
        "SOAP_Note": soap_note
    }
    
    
    output_path = os.path.join(os.path.dirname(__file__), 'output_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2)
    
    print("\n")
    print("RESULTS")
    print("=" * 80)
    
    
    print("\n MEDICAL NLP SUMMARY\n")
    
    print(json.dumps(medical_summary, indent=2))
    
    
    if patient_sentiments:
        print("\n PATIENT SENTIMENT ANALYSIS (Sample)")
        print("-" * 80)
        for i, analysis in enumerate(patient_sentiments[:3], 1):
            print(f"\n{i}. Statement: \"{analysis['Statement'][:80]}...\"")
            print(f"   Sentiment: {analysis['Sentiment']}")
            print(f"   Intent: {analysis['Intent']}")
        if len(patient_sentiments) > 3:
            print(f"\n   ... and {len(patient_sentiments) - 3} more statements")
    
    
    print("\nSOAP NOTE")
    print("-" * 80)
    print(json.dumps(soap_note, indent=2))
    
    print("\n")
    print(f"Complete results saved to: {output_path}")
    


if __name__ == "__main__":
    main()
