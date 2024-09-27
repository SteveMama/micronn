from network import MLPerceptron


xs = [
  [2.0, 3.0, -1.0],
  [3.1, -1.0, 0.5],
  [0.5, 2.0, 1.0],
  [1.0, -0.9, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

x = [2.0, 3.0, -1.0]

model = MLPerceptron(3, [4, 4, 1])

for k in range(20):

    ypred = [model(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    for p in model.parameters():
        p.data += -0.1 * p.grad

    print(k, loss.data)
