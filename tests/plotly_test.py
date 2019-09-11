import plotly.figure_factory as ff
import plotly

z = [[.1, .3, .5, .7, .9],
     [1, .8, .6, .4, .2],
     [.2, 0, .5, .7, .9],
     [.9, .8, .4, .2, 0],
     [.3, .4, .5, .7, 1]]

plotly.io.orca.config.executable = '/home/rico/Downloads/orca'
plotly.io.orca.config.save()
fig = ff.create_annotated_heatmap(z)
fig.write_image("/home/rico/Documents/temp/hm.pdf")
# bytes = fig.to_image(format="png")
fig.show()
