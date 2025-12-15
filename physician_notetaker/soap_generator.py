class SOAPGenerator:
    def __init__(self, nlp_pipeline):
        self.nlp_pipeline = nlp_pipeline

    def generate_soap_note(self, transcript, entities):
        """
        Generates a SOAP note from the transcript and extracted entities.
        """
        
        subjective = {
            "Chief_Complaint": ", ".join(entities.get("Symptoms", [])),
            "History_of_Present_Illness": self._extract_hpi(transcript)
        }

        
        objective = {
            "Physical_Exam": self._extract_physical_exam(transcript),
            "Observations": "Patient appears matching description of recovery." # Placeholder/Inferred
        }

        
        assessment = {
            "Diagnosis": ", ".join(entities.get("Diagnosis", ["Under Investigation"])),
            "Severity": entities.get("Current_Status", "Stable")
        }

        
        plan = {
            "Treatment": ", ".join(entities.get("Treatment", [])),
            "Follow-Up": ", ".join(entities.get("Prognosis", []))
        }

        return {
            "Subjective": subjective,
            "Objective": objective,
            "Assessment": assessment,
            "Plan": plan
        }

    def _extract_hpi(self, transcript):
        """
        Extracts HPI based on patient's initial response.
        """
        
        lines = transcript.split('\n')
        hpi_lines = []
        patient_count = 0
        for line in lines:
            if "Patient:" in line:
                patient_count += 1
                if patient_count <= 2: 
                    hpi_lines.append(line.replace("Patient:", "").strip())
        return " ".join(hpi_lines)

    def _extract_physical_exam(self, transcript):
        """
        Extracts physical exam details.
        """
        
        lower_trans = transcript.lower()
        if "exam" in lower_trans or "check" in lower_trans:
            
            lines = transcript.split('\n')
            exam_details = []
            capturing = False
            for line in lines:
                if "Physical Examination Conducted" in line or "examination" in line.lower():
                    capturing = True
                    continue
                if capturing:
                    if "Doctor:" in line or "Physician:" in line:
                         exam_details.append(line.replace("Doctor:", "").replace("Physician:", "").strip())
                         
                         break 
            
            if exam_details:
                return " ".join(exam_details)
        
        return "General observation conducted."

if __name__ == "__main__":
    
    pass

