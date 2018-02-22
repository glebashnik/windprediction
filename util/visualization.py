from matplotlib import pyplot

def compare_predictions(self, x, y):
    lines = pyplot.plot(x, 'r', y, 'b')
    pyplot.setp(lines, linewidth=0.5)
    pyplot.show()