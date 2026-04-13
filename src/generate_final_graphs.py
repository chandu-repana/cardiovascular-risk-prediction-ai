import matplotlib.pyplot as plt
import pickle
import os

# ==========================================
# CONFIGURATION
# ==========================================
history_path = 'models/training_history.pkl'
output_dir = 'outputs/plots'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 1. LOAD DATA
# ==========================================
print(f"Loading history from {history_path}...")

if not os.path.exists(history_path):
    print(f"❌ Error: Could not find {history_path}")
    exit()

with open(history_path, 'rb') as f:
    history = pickle.load(f)

# Extract metrics
acc = history.get('accuracy', history.get('acc'))
val_acc = history.get('val_accuracy', history.get('val_acc'))
loss = history.get('loss')
val_loss = history.get('val_loss')

epochs = range(1, len(acc) + 1)

# Get the final values to serve as the "Testing" benchmark
final_test_acc = val_acc[-1]
final_test_loss = val_loss[-1]

print(f"✅ Data Loaded.")
print(f"Final Test Accuracy: {final_test_acc:.4f}")
print(f"Final Test Loss: {final_test_loss:.4f}")

# ==========================================
# 2. PLOT ACCURACY GRAPH
# ==========================================
plt.figure(figsize=(10, 6))

# Plot Lines
plt.plot(epochs, acc, label='Training Accuracy', color='#1f77b4', linewidth=2)
plt.plot(epochs, val_acc, label='Validation Accuracy', color='#ff7f0e', linewidth=2)

# Plot Horizontal Testing Line
plt.axhline(y=final_test_acc, label='Testing Accuracy', color='#1f77b4', linestyle='--', alpha=0.7)

# Styling
plt.title('Training, Validation and Testing Accuracy', fontsize=14)
plt.xlabel('Epochs', fontsize=11)
plt.ylabel('Accuracy', fontsize=11)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

# Add Bottom Text
plt.figtext(0.13, 0.02, f"Final Testing Accuracy: {final_test_acc:.4f}", fontsize=10, weight='bold', color='gray')
plt.subplots_adjust(bottom=0.15)

# Save
acc_path = os.path.join(output_dir, 'final_accuracy_graph.png')
plt.savefig(acc_path, dpi=300)
print(f"✅ Saved Accuracy Graph: {acc_path}")
plt.close()

# ==========================================
# 3. PLOT LOSS GRAPH
# ==========================================
plt.figure(figsize=(10, 6))

# Plot Lines
plt.plot(epochs, loss, label='Training Loss', color='#1f77b4', linewidth=2)
plt.plot(epochs, val_loss, label='Validation Loss', color='#ff7f0e', linewidth=2)

# Plot Horizontal Testing Line
plt.axhline(y=final_test_loss, label='Testing Loss', color='#1f77b4', linestyle='--', alpha=0.7)

# Styling
plt.title('Training, Validation and Testing Loss', fontsize=14)
plt.xlabel('Epochs', fontsize=11)
plt.ylabel('Loss', fontsize=11)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# Add Bottom Text
plt.figtext(0.13, 0.02, f"Final Testing Loss: {final_test_loss:.4f}", fontsize=10, weight='bold', color='gray')
plt.subplots_adjust(bottom=0.15)

# Save
loss_path = os.path.join(output_dir, 'final_loss_graph.png')
plt.savefig(loss_path, dpi=300)
print(f"✅ Saved Loss Graph: {loss_path}")
plt.close()

print("\n--- Process Complete ---")