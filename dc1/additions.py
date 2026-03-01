import torch

def compute_sensitivity(model, test_sampler, device, n_classes=6):

    model.eval()
    
    # Initialize counters
    true_positives = torch.zeros(n_classes)
    false_negatives = torch.zeros(n_classes)

    with torch.no_grad():
        for x, y in test_sampler:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            predicted = torch.argmax(outputs, dim=1)

            for cls in range(n_classes):
                # True Positives
                tp = ((predicted == cls) & (y == cls)).sum().item()
                
                # False Negatives
                fn = ((predicted != cls) & (y == cls)).sum().item()

                true_positives[cls] += tp
                false_negatives[cls] += fn

    sensitivity = true_positives / (true_positives + false_negatives + 1e-8)

    return sensitivity


def compute_accuracy(model, sampler, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in sampler:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            predicted = torch.argmax(outputs, dim=1)

            correct += (predicted == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    return accuracy

import torch

def compute_false_negative_rate(model, sampler, device, n_classes=6):
    model.eval()

    true_positives = torch.zeros(n_classes)
    false_negatives = torch.zeros(n_classes)

    with torch.no_grad():
        for x, y in sampler:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            predicted = torch.argmax(outputs, dim=1)

            for cls in range(n_classes):
                tp = ((predicted == cls) & (y == cls)).sum().item()
                fn = ((predicted != cls) & (y == cls)).sum().item()

                true_positives[cls] += tp
                false_negatives[cls] += fn

    fnr = false_negatives / (true_positives + false_negatives + 1e-8)

    return fnr