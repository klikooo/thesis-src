import plotly.graph_objects as go

from plots.cnn.plot_heatmap import generate_annotations


def create_hm(data, x_labels, y_labels,
              color_bar_title="#Parameters", x_title="Stacked layer per conv block", y_title="Kernel size",
              color_scale='Viridis', reverse_scale=True):
    annotations = generate_annotations(data, x_labels, y_labels)

    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=x_labels,
        y=y_labels,
        colorscale=color_scale,
        reversescale=reverse_scale,
        colorbar={"title": color_bar_title}
    ))

    fig.update_layout(
        # title=f'Convergence point L2 {l2_penal}, noise {noise_level}',
        # title='',
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(text=x_title),
            linecolor='black'
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(text=y_title),
            linecolor='black'
        ),
        annotations=annotations,
        margin={
            't': 5,
            'b': 5
        })
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


