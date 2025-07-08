from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from utils.tfidf_with_wordnet import process_tfidf_by_domain as process_with_wordnet
from utils.tfidf_without_wordnet import process_tfidf_by_domain as process_without_wordnet
from utils.clustering_utils import apply_kmeans, apply_dbscan, apply_hierarchical, remove_duplicates, \
    find_optimal_clusters
from utils.visualization_utils import plot_cluster_distribution, plot_dendrogram, plot_elbow_curve
from sklearn.cluster import AgglomerativeClustering

app = Flask(__name__)
app.jinja_env.globals.update(enumerate=enumerate, zip=zip)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global data storage
global_data = {}


@app.route('/')
def index():
    """Render the main upload page"""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle file upload and initial processing"""
    try:
        # Get form data
        threshold = float(request.form['threshold'])
        wordnet_choice = request.form.get('wordnet_choice', 'with')
        file = request.files['file']

        # Validate file type
        if not file.filename.endswith('.csv'):
            return render_template('index.html', error="Please upload a CSV file")

        # Save uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Load and validate data
        df = pd.read_csv(filepath)
        required_columns = ['Input', 'output', 'Domaine']
        missing_cols = [col for col in required_columns if col not in df.columns]

        if missing_cols:
            return render_template('index.html',
                                   error=f"Missing required columns: {', '.join(missing_cols)}")

        # Clean data
        df.dropna(subset=required_columns, inplace=True)

        if df.empty:
            return render_template('index.html',
                                   error="No valid data after removing empty rows")

        # Create processed text column
        df['Processed'] = df.apply(lambda row: f"{row['output']} in {row['Domaine']} domain", axis=1)

        # Store data in global context
        global_data['df'] = df
        global_data['threshold'] = threshold
        global_data['wordnet_choice'] = wordnet_choice

        return redirect(url_for('file_content', threshold=threshold))

    except Exception as e:
        return render_template('index.html', error=f"Error processing file: {str(e)}")


@app.route('/file-content')
def file_content():
    """Display uploaded file content and domains"""
    threshold = float(request.args.get('threshold'))
    df = global_data['df']
    wordnet_choice = global_data.get('wordnet_choice', 'with')

    # Remove the 'Processed' column before displaying
    display_df = df.drop(columns=['Processed'], errors='ignore')

    # Prepare data for display
    domaines = df['Domaine'].unique().tolist()
    html_table = display_df.to_html(classes='data', index=False)
    html_table = "\n".join([line.strip() for line in html_table.splitlines() if line.strip()])

    return render_template('file_content.html',
                         tables=[html_table],
                         domaines=domaines,
                         threshold=threshold,
                         wordnet_choice=wordnet_choice)


@app.route('/domain/<domain>')
def domain_tfidf(domain):
    """Show TF-IDF analysis for a specific domain"""
    df = global_data['df']
    threshold = global_data['threshold']
    wordnet_choice = global_data.get('wordnet_choice', 'with')

    # Filter data for selected domain
    domain_df = df[df['Domaine'] == domain]

    # Use the appropriate TF-IDF processor
    if wordnet_choice == 'with':
        tfidf_matrix, feature_names = process_with_wordnet(domain_df['Processed'].tolist(), threshold)
    else:
        tfidf_matrix, feature_names = process_without_wordnet(domain_df['Processed'].tolist(), threshold)

    tfidf_array = np.array(tfidf_matrix)
    tfidf_filtered = np.where(tfidf_array >= threshold, tfidf_array, 0)

    # Calculate statistics
    stats = {
        'max_value': tfidf_array.max(),
        'min_value': tfidf_array.min(),
        'mean_value': tfidf_array.mean(),
        'non_zero_count': np.count_nonzero(tfidf_filtered),
        'total_elements': tfidf_array.size,
        'non_zero_percentage': (np.count_nonzero(tfidf_filtered) / tfidf_array.size) * 100,
        'total_documents': len(tfidf_array),
        'total_features': len(feature_names)
    }

    # Get top features
    avg_scores = tfidf_filtered.mean(axis=0)
    top_features = sorted(zip(feature_names, avg_scores),
                          key=lambda x: x[1],
                          reverse=True)[:10] if len(feature_names) > 0 else []

    return render_template('domain_tfidf.html',
                           domain=domain,
                           tfidf=tfidf_filtered,
                           features=feature_names,
                           stats=stats,
                           top_features=top_features,
                           threshold=threshold,
                           wordnet_choice=wordnet_choice)


@app.route('/cluster/<domain>')
def cluster_options(domain):
    """Show clustering options for a domain"""
    return render_template('cluster_options.html', domain=domain)


def calculate_hierarchical_optimal_clusters(tfidf_matrix):
    """Calculate optimal number of clusters for hierarchical clustering using multiple methods"""
    n_samples = len(tfidf_matrix)
    max_clusters = min(10, n_samples - 1)  # Reduced max for better performance

    if max_clusters < 2:
        return 2, [0, 0]

    wcss = []
    silhouette_scores = []
    calinski_scores = []

    # Test different cluster numbers
    for k in range(2, max_clusters + 1):  # Start from 2 clusters
        try:
            model = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
            labels = model.fit_predict(tfidf_matrix)

            # Calculate WCSS
            wcss_value = 0
            for i in range(k):
                cluster_points = tfidf_matrix[labels == i]
                if len(cluster_points) > 1:  # Need at least 2 points for meaningful center
                    center = cluster_points.mean(axis=0)
                    wcss_value += np.sum((cluster_points - center) ** 2)
            wcss.append(wcss_value)

            # Calculate silhouette score
            if len(np.unique(labels)) > 1:
                sil_score = silhouette_score(tfidf_matrix, labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)

            # Calculate Calinski-Harabasz score
            if len(np.unique(labels)) > 1:
                ch_score = calinski_harabasz_score(tfidf_matrix, labels)
                calinski_scores.append(ch_score)
            else:
                calinski_scores.append(0)

        except Exception as e:
            wcss.append(0)
            silhouette_scores.append(0)
            calinski_scores.append(0)

    # Find optimal number using multiple criteria
    optimal_k = 2  # Default fallback

    if len(silhouette_scores) > 0:
        # Method 1: Best silhouette score
        best_sil_k = np.argmax(silhouette_scores) + 2

        # Method 2: Elbow method on WCSS
        if len(wcss) > 2:
            # Calculate rate of change
            diff1 = np.diff(wcss)
            diff2 = np.diff(diff1)
            if len(diff2) > 0:
                elbow_k = np.argmax(np.abs(diff2)) + 3  # +3 because we start from k=2
            else:
                elbow_k = best_sil_k
        else:
            elbow_k = best_sil_k

        # Method 3: Balance between silhouette and number of clusters
        # Prefer fewer clusters if silhouette scores are similar
        balanced_k = 2
        for i, score in enumerate(silhouette_scores):
            k = i + 2
            if score > 0.3 and k <= 5:  # Good silhouette and reasonable cluster count
                balanced_k = k
                break

        # Choose the optimal k using a combination of methods
        candidates = [best_sil_k, elbow_k, balanced_k]
        # Remove candidates that are too high
        candidates = [k for k in candidates if k <= min(8, n_samples // 2)]
        if candidates:
            optimal_k = min(candidates)  # Prefer fewer clusters
        else:
            optimal_k = best_sil_k

    # Ensure optimal_k is reasonable
    optimal_k = max(2, min(optimal_k, max_clusters))

    return optimal_k, wcss


@app.route('/cluster/<domain>/<method>')
def cluster_domain(domain, method):
    """Handle clustering requests and visualization"""
    df = global_data['df']
    threshold = global_data['threshold']
    wordnet_choice = global_data.get('wordnet_choice', 'with')

    # Remove duplicates before clustering
    dedup_df = remove_duplicates(df)
    domain_df = dedup_df[dedup_df['Domaine'] == domain]

    if domain_df.empty:
        return f"No data found for domain: {domain}"

    # Use the appropriate TF-IDF processor
    if wordnet_choice == 'with':
        tfidf_matrix, _ = process_with_wordnet(domain_df['Processed'], threshold)
    else:
        tfidf_matrix, _ = process_without_wordnet(domain_df['Processed'], threshold)

    texts = domain_df['Processed'].tolist()

    if method == 'compare':
        results = compare_clustering_algorithms(tfidf_matrix, texts, domain_df)
        comparison_plot = plot_comparison(results)
        return render_template("clustering_result.html",
                               method="COMPARE",
                               domain=domain,
                               comparison_plot=comparison_plot,
                               comparison_results=results,
                               is_comparison=True)

    # Initialize variables
    labels = None
    is_hierarchical = False
    plot = None
    elbow_plot = None

    if method == 'hierarchical':
        # Calculate optimal clusters and WCSS
        optimal_k, wcss = calculate_hierarchical_optimal_clusters(tfidf_matrix)

        # Perform clustering with optimal number of clusters
        labels, n_clusters = apply_hierarchical(tfidf_matrix, n_clusters=optimal_k)
        plot = plot_dendrogram(tfidf_matrix, n_clusters)
        elbow_plot = plot_elbow_curve(wcss)
        is_hierarchical = True

    elif method == 'kmeans':
        labels = apply_kmeans(tfidf_matrix)
        plot = plot_cluster_distribution(labels)
    elif method == 'dbscan':
        try:
            labels = apply_dbscan(tfidf_matrix)
            if len(np.unique(labels)) < 2:
                raise ValueError("DBSCAN found only one cluster")
            plot = plot_cluster_distribution(labels)
        except Exception as e:
            labels = apply_kmeans(tfidf_matrix)
            plot = plot_cluster_distribution(labels)
            error_message = f"DBSCAN failed: {str(e)} - Used KMeans instead"
    else:
        return f"Unknown clustering method: {method}"

    # Calculate cluster stats
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))

    # Prepare all web services in each cluster
    cluster_samples = {}
    for cluster in unique:
        cluster_indices = np.where(labels == cluster)[0]
        cluster_samples[cluster] = [{
            'input': domain_df.iloc[i]['Input'],
            'output': domain_df.iloc[i]['output'],
            'text': texts[i],
            'original_index': domain_df.index[i]
        } for i in cluster_indices]

    # Prepare comparison data with consistent clustering
    comparison_results = get_consistent_comparison_results(tfidf_matrix)

    # Generate comparison plot
    comparison_plot = generate_comparison_plot(comparison_results)

    # Calculate cluster statistics
    cluster_stats = {
        'sizes': cluster_sizes,
        'total': len(labels),
        'silhouette': silhouette_score(tfidf_matrix, labels),
        'calinski_harabasz': calinski_harabasz_score(tfidf_matrix, labels),
        'total_original_services': len(df[df['Domaine'] == domain]),
        'duplicates_removed': len(df[df['Domaine'] == domain]) - len(domain_df)
    }

    if method == 'hierarchical':
        cluster_stats['optimal_clusters'] = optimal_k
        cluster_stats['cutoff_explanation'] = (
            f"Automatically determined optimal number of clusters: {optimal_k} "
            f"using the elbow method on WCSS values"
        )

    return render_template("clustering_result.html",
                           method=method.upper(),
                           domain=domain,
                           plot=plot,
                           elbow_plot=elbow_plot,
                           is_hierarchical=is_hierarchical,
                           cluster_stats=cluster_stats,
                           cluster_samples=cluster_samples,
                           comparison_results=comparison_results,
                           comparison_plot=comparison_plot)


def get_consistent_comparison_results(tfidf_matrix):
    """Get consistent comparison results using the same hierarchical clustering approach"""
    # Calculate optimal clusters for hierarchical
    optimal_k, _ = calculate_hierarchical_optimal_clusters(tfidf_matrix)

    # Get clustering results
    kmeans_labels = apply_kmeans(tfidf_matrix)
    dbscan_labels = apply_dbscan(tfidf_matrix)
    hierarchical_labels, _ = apply_hierarchical(tfidf_matrix, n_clusters=optimal_k)

    return {
        'kmeans': {
            'silhouette': silhouette_score(tfidf_matrix, kmeans_labels),
            'calinski_harabasz': calinski_harabasz_score(tfidf_matrix, kmeans_labels),
            'n_clusters': len(np.unique(kmeans_labels))
        },
        'dbscan': {
            'silhouette': silhouette_score(tfidf_matrix, dbscan_labels),
            'calinski_harabasz': calinski_harabasz_score(tfidf_matrix, dbscan_labels),
            'n_clusters': len(np.unique(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        },
        'hierarchical': {
            'silhouette': silhouette_score(tfidf_matrix, hierarchical_labels),
            'calinski_harabasz': calinski_harabasz_score(tfidf_matrix, hierarchical_labels),
            'n_clusters': len(np.unique(hierarchical_labels))
        }
    }


def compare_clustering_algorithms(X, texts, domain_df):
    """Compare different clustering algorithms with proportionate results"""
    results = {}

    # Calculate optimal clusters for hierarchical clustering
    optimal_k, _ = calculate_hierarchical_optimal_clusters(X)

    # KMeans
    kmeans_labels = apply_kmeans(X)
    results['kmeans'] = {
        'labels': kmeans_labels,
        'silhouette': silhouette_score(X, kmeans_labels),
        'calinski_harabasz': calinski_harabasz_score(X, kmeans_labels),
        'n_clusters': len(np.unique(kmeans_labels)),
        'samples': extract_cluster_samples(X, texts, kmeans_labels, domain_df)
    }

    # DBSCAN
    try:
        dbscan_labels = apply_dbscan(X)
        if len(np.unique(dbscan_labels)) < 2:
            raise ValueError("DBSCAN found only one cluster")
        results['dbscan'] = {
            'labels': dbscan_labels,
            'silhouette': silhouette_score(X, dbscan_labels),
            'calinski_harabasz': calinski_harabasz_score(X, dbscan_labels),
            'n_clusters': len(np.unique(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
            'samples': extract_cluster_samples(X, texts, dbscan_labels, domain_df)
        }
    except Exception as e:
        # Fallback to kmeans results if DBSCAN fails
        results['dbscan'] = results['kmeans'].copy()
        results['dbscan']['error'] = f"DBSCAN failed: {str(e)} - Using KMeans results"

    # Hierarchical - use the same optimal clustering approach
    hierarchical_labels, n_clusters = apply_hierarchical(X, n_clusters=optimal_k)
    results['hierarchical'] = {
        'labels': hierarchical_labels,
        'silhouette': silhouette_score(X, hierarchical_labels),
        'calinski_harabasz': calinski_harabasz_score(X, hierarchical_labels),
        'n_clusters': len(np.unique(hierarchical_labels)),
        'optimal_k': optimal_k,
        'samples': extract_cluster_samples(X, texts, hierarchical_labels, domain_df)
    }

    return results


def extract_cluster_samples(X, texts, labels, domain_df, n_samples=3):
    """Extract sample documents from each cluster with input/output details"""
    clusters = {}
    for cluster in np.unique(labels):
        if cluster == -1:  # Skip noise points in DBSCAN
            continue
        cluster_indices = np.where(labels == cluster)[0]
        clusters[cluster] = [{
            'input': domain_df.iloc[i]['Input'],
            'output': domain_df.iloc[i]['output'],
            'text': texts[i]
        } for i in cluster_indices[:n_samples]]
    return clusters


def plot_comparison(results):
    """Generate comprehensive comparison plot for clustering algorithms"""
    algorithms = list(results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Color scheme
    colors = {'kmeans': '#2E86AB', 'dbscan': '#A23B72', 'hierarchical': '#F18F01'}
    bar_colors = [colors[alg] for alg in algorithms]

    # 1. Silhouette scores
    silhouette_scores = [results[alg]['silhouette'] for alg in algorithms]
    bars1 = axes[0, 0].bar(algorithms, silhouette_scores, color=bar_colors, alpha=0.8)
    axes[0, 0].set_title('Silhouette Score Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Silhouette Score', fontweight='bold')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, score in zip(bars1, silhouette_scores):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

    # 2. Calinski-Harabasz scores
    ch_scores = [results[alg]['calinski_harabasz'] for alg in algorithms]
    bars2 = axes[0, 1].bar(algorithms, ch_scores, color=bar_colors, alpha=0.8)
    axes[0, 1].set_title('Calinski-Harabasz Index', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('CH Index', fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Format CH scores properly
    max_ch = max(ch_scores)
    if max_ch > 1000:
        axes[0, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    for bar, score in zip(bars2, ch_scores):
        height = bar.get_height()
        if score > 1000:
            label = f'{score:.1e}'
        elif score > 100:
            label = f'{score:.0f}'
        else:
            label = f'{score:.2f}'
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2., height + max_ch * 0.02,
                        label, ha='center', va='bottom', fontweight='bold', fontsize=10)

    # 3. Number of clusters
    n_clusters = [results[alg]['n_clusters'] for alg in algorithms]
    bars3 = axes[1, 0].bar(algorithms, n_clusters, color=bar_colors, alpha=0.8)
    axes[1, 0].set_title('Number of Clusters', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Cluster Count', fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)

    for bar, count in zip(bars3, n_clusters):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                        f'{count}', ha='center', va='bottom', fontweight='bold')

    # 4. Combined normalized score
    norm_silhouette = np.array(silhouette_scores)
    if max(ch_scores) > 0:
        norm_ch = np.array(ch_scores) / max(ch_scores)
    else:
        norm_ch = np.zeros(len(ch_scores))
    combined_scores = (norm_silhouette + norm_ch) / 2

    bars4 = axes[1, 1].bar(algorithms, combined_scores, color=bar_colors, alpha=0.8)
    axes[1, 1].set_title('Combined Performance Score', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Normalized Score', fontweight='bold')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(axis='y', alpha=0.3)

    for bar, score in zip(bars4, combined_scores):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

    # Style improvements
    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for tick in ax.get_xticklabels():
            tick.set_fontweight('bold')

    plt.tight_layout()
    return fig_to_base64(fig)


def generate_comparison_plot(comparison_results):
    """Generate professional comparison plot with proper formatting"""
    algorithms = list(comparison_results.keys())

    # Create figure with better styling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Color scheme
    colors = {'kmeans': '#2E86AB', 'dbscan': '#A23B72', 'hierarchical': '#F18F01'}

    # Plot 1: Silhouette Scores
    silhouette_scores = [comparison_results[alg]['silhouette'] for alg in algorithms]
    bars1 = ax1.bar(algorithms, silhouette_scores,
                    color=[colors[alg] for alg in algorithms], alpha=0.8)
    ax1.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax1.set_title('Silhouette Score Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, score in zip(bars1, silhouette_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

    # Plot 2: Calinski-Harabasz Scores (formatted properly)
    ch_scores = [comparison_results[alg]['calinski_harabasz'] for alg in algorithms]
    bars2 = ax2.bar(algorithms, ch_scores,
                    color=[colors[alg] for alg in algorithms], alpha=0.8)
    ax2.set_ylabel('Calinski-Harabasz Index', fontsize=12, fontweight='bold')
    ax2.set_title('Calinski-Harabasz Index Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Format CH scores properly
    max_ch = max(ch_scores)
    if max_ch > 1000:
        # Use scientific notation for very large numbers
        ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    # Add value labels on bars with proper formatting
    for bar, score in zip(bars2, ch_scores):
        height = bar.get_height()
        if score > 1000:
            label = f'{score:.1e}'  # Scientific notation
        elif score > 100:
            label = f'{score:.0f}'  # No decimals for large numbers
        else:
            label = f'{score:.2f}'  # Two decimals for smaller numbers

        ax2.text(bar.get_x() + bar.get_width() / 2., height + max_ch * 0.01,
                 label, ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Add cluster count information
    for i, alg in enumerate(algorithms):
        n_clusters = comparison_results[alg]['n_clusters']
        # Add cluster count below algorithm name
        ax1.text(i, -0.08, f'({n_clusters} clusters)', ha='center', va='top',
                 transform=ax1.get_xaxis_transform(), fontsize=10, style='italic')
        ax2.text(i, -0.08, f'({n_clusters} clusters)', ha='center', va='top',
                 transform=ax2.get_xaxis_transform(), fontsize=10, style='italic')

    # Style improvements
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Clustering Algorithm', fontsize=12, fontweight='bold')
        for tick in ax.get_xticklabels():
            tick.set_fontweight('bold')

    plt.tight_layout()
    return fig_to_base64(fig)


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


if __name__ == '__main__':
    app.run(debug=True)