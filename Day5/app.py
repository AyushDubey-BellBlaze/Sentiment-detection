from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import joblib, numpy as np

app = FastAPI(title="27-Emotion Analyzer")

pipeline = joblib.load("emotion_model_27.pkl")

EMOJIS = {
    "anger":"😠","joy":"😄","sadness":"😢","fear":"😨","surprise":"😲",
    "disgust":"🤢","neutral":"😐","anxiety":"😰","love":"❤️","excitement":"🎉",
    "pride":"💪","shame":"😳","guilt":"😖","envy":"😒","gratitude":"🙏",
    "hope":"🌟","loneliness":"🥺","confusion":"😕","relief":"😮‍💨","contempt":"😏",
    "boredom":"😑","nostalgia":"🌅","jealousy":"💚","frustration":"😤",
    "awe":"🤩","embarrassment":"😳","grief":"😭"
}

# ── Schemas ───────────────────────────────────────────────────────────────────

class TextInput(BaseModel):
    text: str

class EmotionScore(BaseModel):
    label: str
    score: float

class EmotionOutput(BaseModel):
    input      : str
    emotion    : str
    confidence : float
    top_3      : List[EmotionScore]

# ── Helper ────────────────────────────────────────────────────────────────────

def predict(text: str):
    probs   = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_
    pairs   = sorted(zip(classes, probs), key=lambda x: -x[1])
    return pairs

# ── API Endpoint (Postman / code) ─────────────────────────────────────────────

@app.post("/analyze", response_model=EmotionOutput)
def analyze(body: TextInput):
    pairs = predict(body.text)
    return EmotionOutput(
        input      = body.text,
        emotion    = pairs[0][0],
        confidence = round(float(pairs[0][1]), 4),
        top_3      = [EmotionScore(label=c, score=round(float(p), 4)) for c, p in pairs[:3]]
    )

# ── Browser Form (type sentences directly) ────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
  <title>27-Emotion Analyzer</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #f4f6f9; display: flex;
           justify-content: center; padding: 40px 16px; }
    .card { background: white; border-radius: 14px; padding: 36px; width: 100%;
            max-width: 620px; box-shadow: 0 4px 24px rgba(0,0,0,0.08); }
    h1 { font-size: 22px; color: #1a1a2e; margin-bottom: 4px; }
    .sub { font-size: 13px; color: #888; margin-bottom: 28px; }
    label { font-size: 13px; font-weight: 600; color: #444; display: block; margin-bottom: 8px; }
    textarea { width: 100%; padding: 12px 14px; border: 1.5px solid #e0e0e0; border-radius: 10px;
               font-size: 15px; resize: vertical; min-height: 90px; outline: none;
               font-family: inherit; transition: border 0.2s; }
    textarea:focus { border-color: #6c63ff; }
    button { margin-top: 14px; width: 100%; padding: 13px; background: #6c63ff;
             color: white; border: none; border-radius: 10px; font-size: 15px;
             font-weight: 600; cursor: pointer; transition: background 0.2s; }
    button:hover { background: #574fd6; }
    .result { margin-top: 28px; display: none; }
    .top-emotion { background: #f0eeff; border-radius: 12px; padding: 18px 22px;
                   margin-bottom: 18px; display: flex; align-items: center; gap: 14px; }
    .emoji { font-size: 36px; }
    .emo-name { font-size: 22px; font-weight: 700; color: #1a1a2e; text-transform: capitalize; }
    .emo-conf { font-size: 13px; color: #6c63ff; font-weight: 600; margin-top: 2px; }
    .section-title { font-size: 13px; font-weight: 600; color: #888;
                     margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.05em; }
    .bar-row { margin-bottom: 10px; }
    .bar-label { display: flex; justify-content: space-between; margin-bottom: 4px; }
    .bar-name { font-size: 13px; color: #333; text-transform: capitalize; font-weight: 500; }
    .bar-score { font-size: 13px; color: #6c63ff; font-weight: 600; }
    .bar-track { background: #f0eeff; border-radius: 99px; height: 8px; overflow: hidden; }
    .bar-fill  { height: 100%; background: #6c63ff; border-radius: 99px;
                 transition: width 0.6s ease; }
    .input-echo { font-size: 13px; color: #888; margin-bottom: 14px; }
    .input-echo span { color: #333; font-weight: 500; }
    .samples { margin-top: 20px; }
    .samples p { font-size: 12px; color: #aaa; margin-bottom: 8px; }
    .chip { display: inline-block; padding: 5px 12px; background: #f4f6f9; border: 1px solid #e0e0e0;
            border-radius: 99px; font-size: 12px; color: #555; cursor: pointer;
            margin: 3px 3px 3px 0; transition: all 0.2s; }
    .chip:hover { background: #f0eeff; border-color: #6c63ff; color: #6c63ff; }
  </style>
</head>
<body>
<div class="card">
  <h1>🧠 27-Emotion Analyzer</h1>
  <p class="sub">Type any sentence — detects 27 human emotions</p>

  <form id="form">
    <label>Enter your sentence</label>
    <textarea id="txt" placeholder="e.g.  I hate love story&#10;      I am so proud of myself&#10;      I miss those childhood days"></textarea>
    <button type="submit">Analyze Emotion →</button>
  </form>

  <div class="samples">
    <p>Try these examples:</p>
    <span class="chip" onclick="fill('I hate love story')">I hate love story</span>
    <span class="chip" onclick="fill('I love you so much')">I love you so much</span>
    <span class="chip" onclick="fill('I am terrified of tomorrow')">I am terrified</span>
    <span class="chip" onclick="fill('I miss my childhood days')">I miss my childhood</span>
    <span class="chip" onclick="fill('I am so proud of myself')">I am so proud</span>
    <span class="chip" onclick="fill('nothing is working i am so frustrated')">I am frustrated</span>
    <span class="chip" onclick="fill('i feel so alone and forgotten')">I feel alone</span>
    <span class="chip" onclick="fill('thank you for everything')">Thank you</span>
    <span class="chip" onclick="fill('this is breathtaking and beautiful')">This is breathtaking</span>
    <span class="chip" onclick="fill('i cannot believe this just happened')">I cannot believe this</span>
  </div>

  <div class="result" id="result">
    <p class="input-echo">Result for: <span id="r-input"></span></p>
    <div class="top-emotion">
      <div class="emoji" id="r-emoji"></div>
      <div>
        <div class="emo-name" id="r-name"></div>
        <div class="emo-conf" id="r-conf"></div>
      </div>
    </div>
    <p class="section-title">Top 3 Possible Emotions</p>
    <div id="r-bars"></div>
  </div>
</div>

<script>
  function fill(text) {
    document.getElementById('txt').value = text;
    document.getElementById('form').dispatchEvent(new Event('submit'));
  }

  const EMOJIS = {
    anger:"😠",joy:"😄",sadness:"😢",fear:"😨",surprise:"😲",
    disgust:"🤢",neutral:"😐",anxiety:"😰",love:"❤️",excitement:"🎉",
    pride:"💪",shame:"😳",guilt:"😖",envy:"😒",gratitude:"🙏",
    hope:"🌟",loneliness:"🥺",confusion:"😕",relief:"😮",contempt:"😏",
    boredom:"😑",nostalgia:"🌅",jealousy:"💚",frustration:"😤",
    awe:"🤩",embarrassment:"😳",grief:"😭"
  };

  document.getElementById('form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = document.getElementById('txt').value.trim();
    if (!text) return;

    const btn = document.querySelector('button');
    btn.textContent = 'Analyzing...';
    btn.disabled = true;

    const res  = await fetch('/analyze', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text})
    });
    const data = await res.json();

    document.getElementById('r-input').textContent = '"' + data.input + '"';
    document.getElementById('r-emoji').textContent = EMOJIS[data.emotion] || '🎭';
    document.getElementById('r-name').textContent  = data.emotion;
    document.getElementById('r-conf').textContent  = 'Confidence: ' + (data.confidence * 100).toFixed(1) + '%';

    const barsDiv = document.getElementById('r-bars');
    barsDiv.innerHTML = '';
    const maxScore = data.top_3[0].score;

    data.top_3.forEach(item => {
      const pct = (item.score * 100).toFixed(1);
      const w   = ((item.score / maxScore) * 100).toFixed(1);
      barsDiv.innerHTML += `
        <div class="bar-row">
          <div class="bar-label">
            <span class="bar-name">${EMOJIS[item.label] || '🎭'} ${item.label}</span>
            <span class="bar-score">${pct}%</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill" style="width:${w}%"></div>
          </div>
        </div>`;
    });

    document.getElementById('result').style.display = 'block';
    btn.textContent = 'Analyze Emotion →';
    btn.disabled = false;
  });
</script>
</body>
</html>
"""

@app.get("/emotions")
def list_emotions():
    return {
        "total"      : len(pipeline.classes_),
        "emotions"   : list(pipeline.classes_),
        "original_7" : ["surprise","sadness","anger","joy","fear","neutral","disgust"],
        "new_20"     : ["anxiety","love","excitement","pride","shame","guilt","envy",
                        "gratitude","hope","loneliness","confusion","relief","contempt",
                        "boredom","nostalgia","jealousy","frustration","awe","embarrassment","grief"]
    }