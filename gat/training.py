import torch
from sklearn import metrics
import utils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, train_loader):
    model.to(device)
    loss_op = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    model.train()

    total_loss = 0
    for data in train_loader:
        num_graphs = data.num_graphs
        data.batch = None
        data = data.to(device)
        optimizer.zero_grad()
        loss = loss_op(model(data.x, data.edge_index), data.y)
        total_loss += loss.item() * num_graphs
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader.dataset)


def test(model, valid_loader):
    model.eval()

    total_micro_f1 = 0
    for data in valid_loader:
        with torch.no_grad():
            out = model(data.x.to(device), data.edge_index.to(device))
        pred = out.float().cpu() > 0
        #pred = utils.a_third_law(data.y,pred)
        micro_f1 = metrics.f1_score(data.y, pred, average='micro')
        total_micro_f1 += micro_f1 * data.num_graphs
    return total_micro_f1 / len(valid_loader.dataset)
