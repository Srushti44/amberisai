import sys, os
sys.path.insert(0, r'D:\kal ki hackathon')
from audio_module.audio_predictor import AudioPredictor
p=AudioPredictor.get_instance(model_dir=r'D:\kal ki hackathon\audio_module\models')
print('loaded', p.is_loaded)
try:
    res = p.predict(r'D:\kal ki hackathon\audio_module\data\discomfort\10A40438-09AA-4A21-83B4-8119F03F7A11-1430925142-1.0-f-26-dc.wav')
    print(res)
except Exception as e:
    print('prediction failed', e)
