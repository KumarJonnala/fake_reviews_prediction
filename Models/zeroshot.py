import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.chat_models import ChatOllama
from sklearn.metrics import accuracy_score, classification_report

# Output schema
class ReviewClassification(BaseModel):
    label: str = Field(description="fake or original")
    explanation: str

class ZeroShotReviewClassifier:
    def __init__(self, model_name="gemma3:4b"):
        # Initialize the LLM
        self.llm = ChatOllama(model=model_name)
        
        # Prompt template
        self.prompt = PromptTemplate.from_template("""
You are an expert in detecting fake, manipulated, or AI-generated reviews.

Classify the following review strictly as one of:
- "fake"
- "original"

Return ONLY valid JSON:
{{
  "label": "fake/original",
  "explanation": "short explanation"
}}

Review:
"{text}"
""")
        
        # Parser
        self.parser = JsonOutputParser(pydantic_object=ReviewClassification)
        
        # Build the chain
        self.chain = self.prompt | self.llm | self.parser

    # def balance_dataset(self, df):
    #     # Map 4-class labels to binary
    #     df["binary_label"] = df["label"].map({
    #         "truthful": "original",
    #         "OR": "original",
    #         "deceptive": "fake",
    #         "CG": "fake"
    #     })
    #     return df

    def predict(self, texts):
        predictions = []
        explanations = []
        for review in texts:
            try:
                out = self.chain.invoke({"text": review})
                predictions.append(out.label)
                explanations.append(out.explanation)
            except Exception as e:
                predictions.append("error")
                explanations.append(str(e))
        return predictions, explanations

    def evaluate(self, y_true, y_pred):
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print(classification_report(y_true, y_pred))

    def run_pipeline(self, csv_path, output_csv="zero_shot_results_class.csv"):
        # Load dataset
        df = pd.read_csv(csv_path)
        
        # Balance and map labels
        # df = self.balance_dataset(df)
        
        # Predict
        predictions, explanations = self.predict(df["text"])
        df["predicted_label"] = predictions
        df["explanation"] = explanations
        
        # Evaluate
        self.evaluate(df["binary_label"], df["predicted_label"])
        
        # Save results
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
        return df

    def test_single_review(self, review_text):
        # Test a single review
        result = self.chain.invoke({"text": review_text})
        print("Single Review Prediction:")
        print(result)
        return result



# Call the pipeline

if __name__ == "__main__":
    classifier = ZeroShotReviewClassifier(model_name="gemma3:4b")
    
    # Test a single review first
    test_review = ("Stayed at Fairmont Chicago Millennium Park and it was not a good experience. "
                   "The room had been cleaned in a haphazardly fashion and was not pleasant to walk into. "
                   "When I called the service desk for more towels, it took an exceedingly long time "
                   "and the staff was rude. Will not be staying at Fairmont Chicago Millennium Park again.")
    classifier.test_single_review(test_review)
    
    # Run full pipeline on CSV
    df_results = classifier.run_pipeline("../Datasets/merged_reviews.csv")
