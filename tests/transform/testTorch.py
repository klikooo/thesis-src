import torch

batch_size = 4
neurons = 2
centroids = torch.randn((batch_size, neurons))

alpha = torch.randn((batch_size,))
beta = torch.randn((batch_size, ))
# a = a.t()

print(centroids.size())
print(alpha)


alpha = alpha.expand((neurons, batch_size))
beta = beta.expand((neurons, batch_size))
print(alpha)
print(centroids.t())
z = (alpha * centroids.t()).t()
print(z)
print(beta)
z = z + beta.t()
print(z)

print(z[:, 0])
print(z[:, 1])

print("Comparing")
compare = z[:,0].expand(neurons, batch_size).t()
print(compare)
print(z)
x = z <= compare
print(x)
# print((a * centroids.t()))
# torch.bmm(centroids, a)
