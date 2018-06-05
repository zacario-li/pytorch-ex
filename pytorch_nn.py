import torch

N, D_in, H, D_out = 64,1000,100,10
x = torch.randn(N,D_in)
y = torch.randn(N,D_out)

model = torch.nn.Sequential(torch.nn.Linear(D_in,H),
                            torch.nn.ReLU(),
                            torch.nn.Linear(H,D_out))

loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4
for t in range(50000):
    y_pred = model(x)
    loss = loss_fn(y_pred,y)
    print(t,loss.item())

    model.zero_grad()
    loss.backward()

    with torch.no_grad():#这是非常底层的更新parameters的方法，实际项目中，我们直接使用已有的优化器，不需要自己写梯度过程
        for param in model.parameters():
            param -= learning_rate * param.grad