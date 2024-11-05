import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


data = pd.read_csv('sherlock_holmes_text_with_labels.csv')
texts = data['Text']
labels = pd.get_dummies(data[['Label1', 'Label2']])

vectorizer = TfidfVectorizer()
text_vectors = vectorizer.fit_transform(texts)
X = text_vectors.toarray()
y = labels.values

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

classes = labels.columns.tolist()


class ImprovedNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super(ImprovedNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim3, output_dim)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x

input_dim = X.shape[1]
hidden_dim1 = 256
hidden_dim2 = 128
hidden_dim3 = 64
output_dim = y.shape[1]

model = ImprovedNN(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)


num_epochs = 200
patience = 10
best_loss = float('inf')
counter = 0

for epoch in range(num_epochs):
    epoch_loss = 0
    for inputs, labels in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    epoch_loss /= len(dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        counter = 0
        torch.save(model.state_dict(), "model.pth")  # Сохраняем лучшую модель
    else:
        counter += 1
        if counter >= patience:
            print("Раннее завершение обучения")
            break


loaded_model = ImprovedNN(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim)
loaded_model.load_state_dict(torch.load("model.pth"))
loaded_model.eval()


def calculate_accuracy(model, X_tensor, y_tensor, threshold=0.4):  # Попробуйте разные пороги
    model.eval()
    correct_predictions = 0
    total_predictions = y_tensor.shape[0]
    
    with torch.no_grad():
        predictions = model(X_tensor)
        predicted_labels = (predictions > threshold).int()
        
        for i in range(total_predictions):
            if torch.equal(predicted_labels[i], y_tensor[i].int()):
                correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions * 100
    return accuracy

accuracy = calculate_accuracy(loaded_model, X_tensor, y_tensor)
print(f"Точность модели: {accuracy:.2f}%")


test_index = 0  # Индекс примера для предсказания
sample_text_vector = X_tensor[test_index].unsqueeze(0)
actual_labels = y_tensor[test_index].int()  # Фактические метки

with torch.no_grad():
    prediction = loaded_model(sample_text_vector)
    predicted_labels = (prediction > 0.4).int()  # Порог 0.4 для бинарной классификации

predicted_classes = [classes[i] for i, val in enumerate(predicted_labels[0]) if val == 1]
actual_classes = [classes[i] for i, val in enumerate(actual_labels) if val == 1]

print(f'Predicted: "{", ".join(predicted_classes)}", Actual: "{", ".join(actual_classes)}"')
