import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession('kaggle_model/birdaves-biox-base.onnx', providers=['CPUExecutionProvider'])
print('Inputs:')
for i in sess.get_inputs():
    print(f'  {i.name} {i.shape} {i.type}')
print('Outputs:')
for o in sess.get_outputs():
    print(f'  {o.name} {o.shape} {o.type}')

# Try forward at 16kHz mono raw waveform
x = np.random.randn(1, 16000 * 5).astype(np.float32)
print(f'\nInput shape: {x.shape}')
out = sess.run(None, {sess.get_inputs()[0].name: x})
for i, o in enumerate(out):
    print(f'  [{i}]: shape {o.shape} dtype {o.dtype}')
