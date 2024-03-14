import torch
import torchvision.models as models
import time



model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
model.eval()
data = torch.rand(128, 3, 224, 224)

#################### code changes ####################  # noqa F401
import intel_extension_for_pytorch as ipex
model = ipex.optimize(model, dtype=torch.bfloat16)
######################################################  # noqa F401
start = time.time()
with torch.no_grad(), torch.cpu.amp.autocast():
    model(data)

print(f"Elapsed time: {(time.time() - start) * 1000:.2f} ms")
print("Execution finished")