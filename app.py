from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from fastapi.responses import HTMLResponse

#model ve tf-idf yükleme
with open('models/sentiment_model_tfidf.pkl.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
    
#fastp api
app = FastAPI(
    title = "Sentiment Analysis API",
    description = "API for sentiment analysis using a pre-trained model",
    version = "1.0.0"
    
)

#request body
class Review(BaseModel):
    review: str
    
    
#endpoint
@app.post("/predict")
def predict_sentiment(item: Review):
    text = pd.Series([item.review])
    text_vec = vectorizer.transform(text)

    prediction = model.predict(text_vec)[0]
    
    return {"sentiment": prediction}


@app.get("/ui", response_class=HTMLResponse)
def custom_ui():
    return """
    <html>
      <head>
        <title>🎬 SentimentScope</title>
        <style>
          body {font-family: Arial, sans-serif; margin: 60px; background-color: #f9fafc; color: #333;}
          h2 {color: #2c3e50;}
          textarea {width: 100%; height: 120px; font-size: 16px; padding: 10px; border-radius: 8px; border: 1px solid #ccc;}
          button {padding: 10px 25px; background: #0078D7; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 15px;}
          button:hover {background: #005fa3;}
          #result {margin-top: 20px; font-size: 18px;}
          footer {margin-top: 50px; color: #777; font-size: 13px;}
        </style>
      </head>
      <body>
        <h2>SentimentScope - Film Yorumu Analiz Aracı</h2>
        <p>Bir film yorumu yazın, modelin ne düşündüğünü görün:</p>
        <textarea id="review" placeholder="örnek: This movie was boring and too long..."></textarea><br><br>
        <button onclick="predict()">Tahmin Et</button>
        <div id="result"></div>
        <footer>TF-IDF + Logistic Regression</footer>

        <script>
          async function predict() {
              const text = document.getElementById('review').value;
              if (!text.trim()) {
                  document.getElementById('result').innerHTML = '<span style="color:red;">Lütfen bir yorum girin.</span>';
                  return;
              }
              const res = await fetch('/predict', {
                  method: 'POST',
                  headers: {'Content-Type': 'application/json'},
                  body: JSON.stringify({'review': text})
              });
              const data = await res.json();
              const color = data.sentiment === 'positive' ? '#27ae60' : '#c0392b';
              document.getElementById('result').innerHTML = 
                  'Sonuç: <b style="color:' + color + ';">' + data.sentiment.toUpperCase() + '</b>';
          }
        </script>
      </body>
    </html>
    """

#root endpoint
@app.get("/")
def root():
    return {"message": "API is working. Use /predict endpoint to analyze sentiment."}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)