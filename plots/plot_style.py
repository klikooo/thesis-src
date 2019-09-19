import matplotlib.style
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm


def plot(n, style):
    plt.figure()
    plt.title(f'Style: {style}')
    matplotlib.style.use(style)
    x = np.array([1, 2, 3, 4])
    for i in range(n):
        plt.plot(x, x + i, label=f'{i}')
    plt.show()


def do():
    styles = matplotlib.style.available
    print(f'Styles: {styles}')
    for style in styles:
        plot(10, style)

# plot(10, "fivethirtyeight")
# do()


def plot2(n):
    x = np.array([1, 2, 3, 4])
    for i in range(n):
        plt.plot(x, x + i, label=f'{i}', c=cm.spring(0))
    plt.show()

plot2(10)
