from src.data_processing import *
from src.model_building import *
from src.visualization import *

# Check if data directory exists
if not os.path.exists(DATA_DIR):
    print(f"Error: Directory '{DATA_DIR}' not found.")
    exit()

# Load dataset using TensorFlow's image dataset utility
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    seed=123,
    shuffle=True,
    image_size=(512, 512),
    batch_size=32
)

# Set image directory
image_dir = DATA_DIR

# Get filepaths and labels
filepaths, labels = get_filepaths_labels(image_dir)
# Create DataFrame with filepaths and labels
image_df = create_dataframe(filepaths, labels)

# Split the dataset into training and test sets
train_df, test_df = split_train_test_data(image_df)

# Display class distribution
class_names = dataset.class_names
class_dis = [len(os.listdir(DATA_DIR / name)) for name in class_names]


# Display a bar plot of class distribution
display_class_distribution_barplot(class_names, class_dis)

# Create a transfer learning model (ResNet152V2)
resnet152V2 = create_resnet152V2_model()

# Compile the model
compile_model(resnet152V2)

# Set up callbacks for early stopping and model checkpoint
cbs = setup_callbacks("ResNet152V2")

# Initialize lists to store training/validation accuracy and loss for each fold
train_accs, val_accs, train_losses, val_losses = [], [], [], []

# Initialize list to store fold-wise accuracy
fold_accuracy = []

# Set up KFold cross-validation
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True)

# Loop through folds
for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    train_fold, val_fold = train_df.iloc[train_idx], train_df.iloc[val_idx]

    # Set up image data generators for training and validation
    train_images, val_images = get_image_data_generators(train_fold, val_fold)

    # Train the model
    history = train_model(resnet152V2, train_images, val_images, cbs)

    # Store accuracy and loss history for this fold
    train_accs.append(history.history['accuracy'])
    val_accs.append(history.history['val_accuracy'])
    train_losses.append(history.history['loss'])
    val_losses.append(history.history['val_loss'])

    # Evaluate the model on the current fold
    _, accuracy = evaluate_model(resnet152V2, val_images)
    fold_accuracy.append(accuracy)
    print(f"Accuracy for Fold {fold + 1}: {accuracy}")

# Calculate and print average cross-validation accuracy
avg_accuracy = np.mean(fold_accuracy)
print(f"Average Cross-Validation Accuracy: {avg_accuracy}")

# Plotting training and validation accuracy/loss for the best fold
best_train_acc, best_train_loss = train_accs[4], train_losses[4]
best_val_acc, best_val_loss = val_accs[4], val_losses[4]

epochs = range(1, len(best_train_acc) + 1)
plot_accuracy_loss(epochs, best_train_acc, best_val_acc, best_train_loss, best_val_loss)

# Plotting accuracy across epochs for each fold
plot_accuracy_epochs(train_accs, val_accs)

# Plotting train losses across epochs for each fold
plot_loss_epochs(train_losses)

# Plotting validation accuracy across epochs for each fold
plot_validation_accuracy_epochs(val_accs)

# Plotting validation losses across epochs for each fold
plot_validation_loss_epochs(val_losses)

# Evaluate the model on the test set
test_images = get_image_data_generators(test_df, None)[0]
results = evaluate_model(resnet152V2, test_images)

# Extracting predictions and true labels
y_pred = np.argmax(results, axis=1)
y_true = test_images.classes

# Plotting the confusion matrix
plot_confusion_matrix(y_true, y_pred, class_names)
