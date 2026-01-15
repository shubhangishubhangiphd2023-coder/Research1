from mmsdk import mmdatasdk

# This will download automatically
dataset = mmdatasdk.mmdataset({
    'cmumosei_text': 'cmumosei/text',
    'cmumosei_audio': 'cmumosei/audio',
    'cmumosei_vision': 'cmumosei/vision',
    'cmumosei_labels': 'cmumosei/labels'
})
