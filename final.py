import pandas as pd
from tkinter import *
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Global variable to hold the dataset
data = None

# Function to load CSV file
def load_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        try:
            global data
            data = pd.read_csv(file_path)
            messagebox.showinfo("Success", "File loaded successfully")
            show_columns()  # Display columns in the DataFrame
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

# Function to show available columns in the dataset
def show_columns():
    if data is not None:
        columns = data.columns.tolist()
        target_label.config(state=NORMAL)
        target_label.delete(0, END)
        target_label.insert(0, columns[0])  # Default target is the first column
        feature_listbox.delete(0, END)
        for col in columns:
            feature_listbox.insert(END, col)
        feature_listbox.select_set(0)  # Select the first feature by default

# Function to plot the graph
def plot_graph(y_test, y_pred):
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(5, 4))
    # Plot the data
    unique_labels = sorted(set(y_test))
    counts_actual = [list(y_test).count(label) for label in unique_labels]
    counts_predicted = [list(y_pred).count(label) for label in unique_labels]

    bar_width = 0.35
    x = range(len(unique_labels))
    ax.bar(x, counts_actual, width=bar_width, label="Actual", alpha=0.7)
    ax.bar([i + bar_width for i in x], counts_predicted, width=bar_width, label="Predicted", alpha=0.7)

    ax.set_xticks([i + bar_width / 2 for i in x])
    ax.set_xticklabels(unique_labels)
    ax.set_title("Actual vs Predicted Distribution")
    ax.set_xlabel("Classes")
    ax.set_ylabel("Count")
    ax.legend()

    # Embed the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(pady=10)
    canvas.draw()

# Function to run the model
def run_model():
    try:
        # Ensure data is loaded
        if data is None:
            messagebox.showerror("Error", "Please load a dataset first.")
            return
        
        # Get target column and selected features
        target = target_label.get()
        selected_features = [feature_listbox.get(i) for i in feature_listbox.curselection()]
        
        if target not in data.columns:
            messagebox.showerror("Error", f"Target column '{target}' does not exist.")
            return

        if not selected_features:
            messagebox.showerror("Error", "Please select at least one feature.")
            return

        # Encode the target variable
        label_encoder = LabelEncoder()
        data['impact_encoded'] = label_encoder.fit_transform(data[target])

        X = data[selected_features]
        y = data['impact_encoded']

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=40, stratify=y
        )

        # Hyperparameter tuning for zzz
        param_grid = {
            'hidden_layer_sizes': [(50, 50), (100, 100), (50, 100, 50)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [200, 300]
        }
        mlp_model = MLPClassifier(random_state=42)
        grid_search = GridSearchCV(mlp_model, param_grid, scoring='accuracy', cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Use the best estimator
        best_model = grid_search.best_estimator_

        # Predict on the test set
        y_pred = best_model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

        # Display results in the text box
        result_text.delete(1.0, END)
        result_text.insert(END, f"Best Parameters: {grid_search.best_params_}\n")
        result_text.insert(END, f"Accuracy: {accuracy:.2f}\n")
        result_text.insert(END, "Classification Report:\n")
        result_text.insert(END, report)

        # Plot the graph
        plot_graph(y_test, y_pred)

    except Exception as e:
        messagebox.showerror("Error", f"Error running model: {e}")

# Creating the Tkinter GUI window
root = Tk()
root.title("Oil Spill Impact Prediction - MLP Classifier with Graph")

# File loading section
file_button = Button(root, text="Load Dataset", command=load_file)
file_button.pack(pady=10)

# Target and feature selection section
target_label = Entry(root, state=DISABLED, width=30)
target_label.pack(pady=5)

feature_listbox = Listbox(root, selectmode=MULTIPLE, height=6, width=30)
feature_listbox.pack(pady=5)

# Buttons to run the model
run_button = Button(root, text="Run Model", command=run_model)
run_button.pack(pady=10)

# Text box to display results
result_text = Text(root, width=60, height=15)
result_text.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()
