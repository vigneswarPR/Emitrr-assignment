# Emitrr-assignment
# Physician Notetaker

An AI system for medical transcription, NLP-based summarization, and sentiment analysis of physician-patient conversations. This project extracts medical entities, classifies sentiment, and generates structured SOAP notes from transcripts.

## Setup Instructions

1. Install Python Dependencies

   Run the following command:

   ```bash
   pip install -r requirements.txt
   ```

2. Install Medical NLP Models

   Install the scispacy medical entity recognition model:

   ```bash
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
   ```


3. ```bash
   pip install "transformers[torch]"
   ```
4.To Run
  ```bash
  cd physician_notetaker
  python main.py
  ``` 
main.py - Orchestrates the full pipeline from reading the transcript to saving JSON results

nlp_pipeline.py - Extracts medical entities like diagnosis and treatment using scispacy and regex

sentiment.py - Determines patient emotion and intent using a local transformer model

soap_generator.py - Formats the extracted medical data into a standard SOAP note structure

sample_transcript.txt - Contains the raw dialogue input between the physician and patient

requirements.txt - Lists the Python libraries required to run the project

output_results.json - Stores the final structured output including entities and the SOAP note
