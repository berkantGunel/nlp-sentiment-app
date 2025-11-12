# IMDB Film Yorum Analizi

Bu proje, IMDB film yorumlarını **doğal dil işleme (NLP)** teknikleriyle analiz eder.  
Model, bir yorumun **pozitif** veya **negatif** olduğunu tahmin eder.

Projede kullanılan yöntem:

- TF-IDF tabanlı metin temsil yöntemi
- Logistic Regression sınıflandırıcısı
- FastAPI tabanlı REST API
- HTML tabanlı sade web arayüzü (/ui)

---

## 🚀 Özellikler

✅ IMDB yorum veri setiyle model eğitimi  
✅ CountVectorizer ve TF-IDF karşılaştırması  
✅ Logistic Regression modeliyle sınıflandırma  
✅ FastAPI ile gerçek zamanlı tahmin servisi  
✅ /docs ve /ui arayüzlerinden etkileşimli kullanım

---

## ⚙️ Kurulum

1. Gerekli kütüphaneleri yükle:
   ```bash
   pip install -r requirements.txt
   ```
2. Uygulamayı başlat:
   py app.py
3. Test et:
   http://127.0.0.1:8000/ui
