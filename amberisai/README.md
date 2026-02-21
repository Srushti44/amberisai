# AmberisAI — Phase 1 Flask Backend

## Folder Structure Expected on Disk

```
kal ki hackathon/
├── audio_module/               ← YOUR EXISTING MODULE (untouched)
│   ├── models/
│   ├── audio_predictor.py
│   ├── visualization.py
│   └── ...
├── image_model_updated/        ← YOUR EXISTING MODULE (untouched)
│   ├── app.py
│   ├── keras_model.h5
│   └── labels.txt
└── amberisai/                  ← THIS FLASK APP (new)
    ├── app.py
    ├── config.py
    ├── requirements.txt
    ├── models/
    ├── routes/
    ├── utils/
    ├── static/visuals/
    ├── uploads/
    └── data/
```

---

## Setup & Run

```bash
cd amberisai
pip install -r requirements.txt
python app.py
```

Server runs at: http://localhost:5000

---

## ⚠️ IMPORTANT: Fix function names in routes/predict.py

Open `routes/predict.py` and check what function your modules expose:

**For audio_predictor.py** — look inside audio_module/audio_predictor.py and find the main prediction function name. Then update this block:
```python
# Line ~55 in routes/predict.py
if hasattr(audio_predictor, 'predict'):
    analysis = audio_predictor.predict(file_path)
```
Change `'predict'` to whatever your actual function is named.

**For image app.py** — same thing. Open image_model_updated/app.py, find the prediction function, update accordingly.

---

## curl Test Commands

### 1. Create baby profile
```bash
curl -X POST http://localhost:5000/profile \
  -H "Content-Type: application/json" \
  -d '{"nickname": "BabyA", "age_days": 45, "allergies": ["milk"]}'
```

Expected response:
```json
{"success": true, "baby_id": 1, "nickname": "BabyA", "age_days": 45, "allergies": ["milk"]}
```

---

### 2. Get baby profile
```bash
curl http://localhost:5000/profile/1
```

---

### 3. Predict audio (cry analysis)
```bash
curl -X POST http://localhost:5000/predict-audio \
  -F "file=@/path/to/cry.wav" \
  -F "baby_id=1"
```

Expected response:
```json
{
  "success": true,
  "session_id": "sess_abc123xyz",
  "baby_id": 1,
  "audio_analysis": {
    "primary_condition": "hunger",
    "confidence": 0.92,
    "secondary_condition": "discomfort",
    "secondary_confidence": 0.15,
    "other_conditions": {"belly_pain": 0.08, "tired": 0.03},
    "features": [...]
  },
  "visualization_url": "/static/visuals/sess_abc123xyz_hunger.png",
  "timestamp": "2026-02-20T05:16:00Z"
}
```

---

### 4. Predict image (skin analysis)
```bash
curl -X POST http://localhost:5000/predict-image \
  -F "file=@/path/to/baby_skin.jpg" \
  -F "baby_id=1"
```

Expected response:
```json
{
  "success": true,
  "session_id": "sess_xyz987abc",
  "baby_id": 1,
  "image_analysis": { ... your image module's exact JSON ... },
  "timestamp": "2026-02-20T05:16:00Z"
}
```

---

### 5. View generated PNG visualization
```bash
curl http://localhost:5000/static/visuals/sess_abc123xyz_hunger.png --output result.png
```

---

## Troubleshooting

**"No known predict function found"** → Open your audio_predictor.py or image app.py and find the prediction function name. Update routes/predict.py accordingly.

**Import errors** → The sys.path in routes/predict.py assumes amberisai/ is inside "kal ki hackathon/" alongside audio_module/ and image_model_updated/. If your structure differs, adjust the AUDIO_MODULE_DIR and IMAGE_MODULE_DIR paths in routes/predict.py.

**Visualization not generating** → Check utils/visual_generator.py. It tries multiple function names from your visualization.py. Add a print in visualization.py to confirm the function name, then update visual_generator.py.