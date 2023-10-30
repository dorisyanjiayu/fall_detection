import time
import torch 

from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights




def parameter_count(model):
    count = 0
    for _, param in model.named_parameters():
        count += param.numel()
    return count

def flop_count(model):
    count = 0
    for i, (_, param) in enumerate(model.named_parameters()):
        if i % 2 == 0:
            a, b = param.size()
            count += a * (b + b-1)
            print("this layer:", a *(b+b-1))
    return count

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, sample in enumerate(dataloader): #?
        target = sample['category']
        img = sample['img_list']
        print(target, img)
        print("batch", batch)
        print("istensorimg", torch.is_tensor(img))
        # print("img tensor shape", img.shape, "target shape", target.shape)
        img = img.to(torch.float32)
        print(img)
        pred = model(img)
        loss = loss_fn(pred, torch.Tensor(target))

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(img)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    idx = 0
    with torch.no_grad():
        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        for img, target in dataloader:
            start_t = time.time()
            img = img.to(torch.float32)
            pred = model(img)
            test_loss += loss_fn(pred, target).item()
            correct += (torch.argmax(pred, 1) == target).type(torch.float).sum().item() #?
            #if (idx < 50):
                #idx += 1
                #print("inference for", idx, "th instance is", time.time() - start_t)

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100*correct

def get_Krcnn():
    weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    k_transforms = weights.transforms()
    keypoint_rcnn = keypointrcnn_resnet50_fpn(weights=weights, progress=False)
    keypoint_rcnn.eval()
    return keypoint_rcnn