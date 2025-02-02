import streamlit as st
from transformers import pipeline

st.info("The model used to analyze is Hugging Face's 'sentiment-analysis' utilizing 'distilbert-base-uncased-finetuned-sst-2-english'.", icon="ℹ️")
st.write("Gain an idea about a user's comment or review by using this comment analyzer.")
text = st.text_input("Enter text to analyze")

@st.cache_resource
def get_model():
  return pipeline("sentiment-analysis")

model = get_model()
if text:
  result = model(text)
  if result[0]["label"] == "POSITIVE":
    st.write("We received a positive review!")
  else:
    st.write("We received a negative review.")

  st.write("The confidence score is {:.2%}.".format(result[0]["score"]))

  st.write(topic_model.predict(text))

