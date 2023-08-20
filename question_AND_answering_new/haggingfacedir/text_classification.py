from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("badalsahani/text-classification-multi", use_auth_token=True)

tokenizer = AutoTokenizer.from_pretrained("badalsahani/text-classification-multi", use_auth_token=True)

inputs = tokenizer("I love AutoTrain", return_tensors="pt")

outputs = model(**inputs)