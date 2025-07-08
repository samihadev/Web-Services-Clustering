document.addEventListener("DOMContentLoaded", function () {
    // This code will run once the page content is loaded.

    const pcaData = {
        x: {{ cluster_data.pca_points | tojson | safe }}.map(p => p[0]),
        y: {{ cluster_data.pca_points | tojson | safe }}.map(p => p[1]),
        mode: 'markers',
        type: 'scatter',
        marker: { size: 10, color: {{ cluster_data.labels | tojson | safe }} },
        text: {{ cluster_data.texts | tojson | safe }},
    };

    const tsneData = {
        x: {{ cluster_data.tsne_points | tojson | safe }}.map(p => p[0]),
        y: {{ cluster_data.tsne_points | tojson | safe }}.map(p => p[1]),
        mode: 'markers',
        type: 'scatter',
        marker: { size: 10, color: {{ cluster_data.labels | tojson | safe }} },
        text: {{ cluster_data.texts | tojson | safe }},
    };

    Plotly.newPlot('pca-plot', [pcaData], {
        title: 'PCA Visualization',
        xaxis: { title: 'PCA 1' },
        yaxis: { title: 'PCA 2' },
        height: 500
    });

    Plotly.newPlot('tsne-plot', [tsneData], {
        title: 't-SNE Visualization',
        xaxis: { title: 't-SNE 1' },
        yaxis: { title: 't-SNE 2' },
        height: 500
    });
});
