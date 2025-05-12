import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt



# Define the embedding network
class EmbeddingNet(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(EmbeddingNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, embedding_dim * 2),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x):
        return self.fc(x)

    # Add a method to get L2 regularization term for all parameters
    def get_l2_norm(self):
        l2_norm = torch.tensor(0.0)
        for param in self.parameters():
            l2_norm += torch.norm(param, p=2)**2
        return l2_norm

# Define the Prototypical Network
class PrototypicalNetwork(nn.Module):
    def __init__(self, embedding_net):
        super(PrototypicalNetwork, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, support, query, n_classes, n_support, n_query):
        support_emb = self.embedding_net(support)
        query_emb = self.embedding_net(query)
        
        # Ensure reshaping is valid
        support_emb = support_emb.view(n_classes, n_support // n_classes, -1)
        
        prototypes = support_emb.mean(dim=1)  # Compute class prototypes
        dists = torch.cdist(query_emb, prototypes)  # Compute distances
        return -dists

    # Add a method to get L2 regularization term
    def get_l2_norm(self):
        return self.embedding_net.get_l2_norm()
        


def get_test_accuracy(reduced_data , reduced_test , labels , y_test ,num_classes = 5 , lr=0.05 , num_epoch=1000 , l2_lambda=0.01):
    # Model setup

    # Convert labels to range [0, 4] and ensure correct dtype
    y_train = torch.tensor(labels, dtype=torch.long)  
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Count samples per class
    class_indices = {c: np.where(y_train.numpy() == c)[0] for c in range(num_classes)}

    # Determine equal support set size for all classes
    n_support_per_class = min(len(class_indices[c]) * 2 // 3 for c in range(num_classes))
    n_support_per_class -= n_support_per_class % 2  # Ensure even number
    n_support = n_support_per_class * num_classes  # Total support samples

    support_indices = []
    query_indices = []

    for c in range(num_classes):
        support_samples = np.random.choice(class_indices[c], n_support_per_class, replace=False)
        query_samples = np.setdiff1d(class_indices[c], support_samples)

        support_indices.extend(support_samples)
        query_indices.extend(query_samples)

    # Convert to tensors
    support_set = torch.tensor(reduced_data[support_indices], dtype=torch.float32)
    query_set = torch.tensor(reduced_data[query_indices], dtype=torch.float32)
    support_labels = torch.tensor(y_train[support_indices], dtype=torch.long)
    query_labels = torch.tensor(y_train[query_indices], dtype=torch.long)
    input_dim = support_set.shape[1]  
    embedding_dim = 50
    embedding_net = EmbeddingNet(input_dim, embedding_dim)
    proto_net = PrototypicalNetwork(embedding_net)

    # Training setup
    optimizer = optim.Adam(proto_net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Define class count and sample sizes
    n_classes = num_classes
    n_query = len(query_indices)  # Total query samples

    losses  = []

    # Training loop
    for epoch in range(num_epoch):
        proto_net.train()
    
        logits = proto_net(support_set, query_set, n_classes, n_support, n_query)
        loss = criterion(logits, query_labels)
        l2_reg = proto_net.get_l2_norm()
        loss += l2_lambda * l2_reg
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    losses = np.array(losses)
    plt.figure(figsize=(10, 5))
    plt.plot(losses)

    # Evaluate on test set
    proto_net.eval()
    X_test_tensor = torch.tensor(reduced_test, dtype=torch.float32)  # Convert NumPy array to tensor
    test_logits = proto_net(support_set, X_test_tensor, n_classes, n_support, X_test_tensor.shape[0])
    test_preds = torch.argmax(test_logits, dim=1)
    accuracy = (test_preds == y_test).float().mean().item()
    return accuracy