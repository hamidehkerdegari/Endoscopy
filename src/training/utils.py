# import torch

# def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}

#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0
        
#         for inputs, labels in train_loader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#         train_loss = running_loss / len(train_loader)
#         train_accuracy = correct / total

#         history['train_loss'].append(train_loss)
#         history['train_accuracy'].append(train_accuracy)

#         # Validation step
#         model.eval()
#         val_loss = 0.0
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for inputs, labels in val_loader:
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item()
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()

#         val_loss = val_loss / len(val_loader)
#         val_accuracy = correct / total

#         history['val_loss'].append(val_loss)
#         history['val_accuracy'].append(val_accuracy)

#         print(f'Epoch {epoch+1}/{num_epochs}, '
#               f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
#               f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

#     return model, history  # Returning both the trained model and history

import torch

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            labels = labels.view(-1, 1).float()  # Ensure labels are [batch_size, 1]

            optimizer.zero_grad()
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # For binary classification, apply a threshold to outputs
            predicted = (outputs >= 0.5).float()  # Use threshold 0.5 to get binary predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total

        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)

        # Validation step
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Convert labels to float
                labels = labels.view(-1, 1).float()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                predicted = (outputs >= 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total

        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    return model, history
