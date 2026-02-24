import torch

checkpoint = torch.load('best_model.pth', map_location='cpu', weights_only=False)
print("Checkpoint keys:", checkpoint.keys())

if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

# Print first few keys to see structure
print("\nFirst 10 keys in state_dict:")
for i, key in enumerate(list(state_dict.keys())[:10]):
    print(f"{key}: {state_dict[key].shape}")

# Look for classifier layers
print("\nClassifier layers:")
for key in state_dict.keys():
    if 'classifier' in key:
        print(f"{key}: {state_dict[key].shape}")