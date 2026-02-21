import h5py
import json

path = r'D:\kal ki hackathon\image_model_updated\keras_model.h5'

with h5py.File(path, 'r+') as f:
    config = f.attrs['model_config']
    if isinstance(config, bytes):
        config = config.decode('utf-8')
    
    # Parse JSON and remove 'groups' key recursively
    config_dict = json.loads(config)
    
    def remove_groups(obj):
        if isinstance(obj, dict):
            obj.pop('groups', None)
            for v in obj.values():
                remove_groups(v)
        elif isinstance(obj, list):
            for item in obj:
                remove_groups(item)
    
    remove_groups(config_dict)
    
    fixed_config = json.dumps(config_dict)
    f.attrs['model_config'] = fixed_config.encode('utf-8')
    print('Done! groups param removed from model config.')
    print('Model file patched successfully.')