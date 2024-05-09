import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from wordcloud import WordCloud
from textblob import TextBlob
import numpy as np
from textblob import TextBlob
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import altair as alt



# Load your dataset
file_path = "C:/Studies/DV/DV_LAB/coffee/coffee.csv"
data = pd.read_csv(file_path)

# Function to set custom CSS style
def set_custom_css():
    st.markdown(
    """
    <style>
    .reportview-container {
        background: #222831; /* Black background */
        color: red; /* White text color */
    }
    .sidebar .sidebar-content {
        background: #000000; /* Black background for sidebar */
        color: red; /* White text color for sidebar */
    }
    .sidebar .sidebar-content .block-container {
        color: red; /* White text color for sidebar content */
    }
    .Widget>label {
        color: red; /* White text color for labels in widgets */
    }
    .stRadio label span {
        color: red; /* White text color for radio button labels */
    }
    .stMarkdown p {
        color: red; /* Red text color for markdown paragraphs */
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: red; /* Red text color for markdown headings */
    }
    .navbar {
        overflow: hidden;
        background-color: #333;
        position: fixed;
        top: 0;
        width: 100%;
    }

    .navbar a {
        float: left;
        display: block;
        color: #f2f2f2;
        text-align: center;
        padding: 14px 20px;
        text-decoration: none;
    }

    .navbar a:hover {
        background-color: #ddd;
        color: black;
    }

    .navbar a.active {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """,
        unsafe_allow_html=True
    )

# Function to set custom page styling
def set_page_style(page_name):
    st.markdown(
    f"""
    <style>
    .reportview-container .main .block-container h1 {{
        color: red; /* Red subtitle color */
    }}
    .reportview-container .main .block-container h2 {{
        color: red; /* Red subtitle color */
    }}
    .reportview-container .main .block-container h3 {{
        color: red; /* Red subtitle color */
    }}
    .reportview-container .main .block-container h4 {{
        color: red; /* Red subtitle color */
    }}
    .reportview-container .main .block-container h5 {{
        color: red; /* Red subtitle color */
    }}
    .reportview-container .main .block-container h6 {{
        color: red; /* Red subtitle color */
    }}
    </style>
    """,
        unsafe_allow_html=True
    )
    
# Add a custom JavaScript function to handle page navigation
st.markdown(
"""
<script>
  function navigate(page) {
    const url = new URL(window.location);
    url.hash = page;
    window.location.href = url;
  }
</script>
""",
unsafe_allow_html=True
)


# Function for top navigation bar
# Function for top navigation bar
def top_navigation_bar():
    st.markdown(
    """
    <div class="navbar">
      <h1 style="text-align: left; color: white; padding-top: 50px;">☕ COFFEE REVIEW ANALYSIS</h1>
      <p style="text-align: left; color: white; font-size: 15px; padding-left: 20px;">Exploring the analysis of coffee reviews by using table analysis, multivariate analysis, and text analysis.</p>
    </div>
    """,
    unsafe_allow_html=True
    )

# Function to handle page navigation
def handle_page_navigation():
    st.write("")  # Add empty space for separation
    st.write("")
    st.write("")  # Add empty space for separation
    st.write("")
    if st.button("📊 Table Analysis"):
        st.session_state.page = "Table_Analysis"
    if st.button("📝 Text Analysis"):
        st.session_state.page = "Text_Analysis"
    if st.button("📈 Multivariate Analysis"):
        st.session_state.page = "Multivariate_Analysis"
    





# Rest of the code remains the same

def explore_data():
    st.title("Table Analysis")
    st.markdown("### Display Data")
    display_data()
    st.markdown("### Contingency Table")
    display_contingency_table()
    st.markdown("### Tree Table")
    display_tree_table()
    st.markdown("### Fish-eye Distortion")
    fish_eye_distortion()
    

def display_data():
    st.subheader("Display Data")

    # Define a list of attributes
    attributes = ['name', 'roaster', 'roast', 'loc_country', 'origin_1', 'origin_2', '100g_USD', 'rating', 'review_date', 'desc_1', 'desc_2', 'desc_3']

    # Allow users to select attributes to display
    selected_attributes = st.multiselect("Select Attributes", attributes)

    # Check if any attributes are selected
    if not selected_attributes:
        st.write("Please select attributes to display.")
        return

    # Display the selected attributes as a table
    st.dataframe(data[selected_attributes])





import plotly.figure_factory as ff

def display_contingency_table():
    columns = st.multiselect("Select columns for contingency table", data.columns)

    # Check if at least two columns are selected
    if len(columns) < 2:
        st.write("Please select at least two columns.")
        return

    # Create contingency table
    try:
        contingency_table = pd.crosstab(data[columns[0]], data[columns[1]])
        fig = ff.create_annotated_heatmap(
            z=contingency_table.values,
            x=contingency_table.columns.tolist(),
            y=contingency_table.index.tolist(),
            colorscale='Viridis'
        )
        fig.update_layout(title='Contingency Table', xaxis_title=columns[1], yaxis_title=columns[0])
        st.plotly_chart(fig)
    except Exception as e:
        st.write(f"Error: {e}")


def display_tree_table():
    tree_data = generate_tree_data(data)
    
    # Display a dropdown selection for choosing the tree data
    selected_tree = st.selectbox("Select Tree Data:", list(tree_data.keys()))

    # Display the selected tree data
    st.write(tree_data[selected_tree])


def generate_sparklines(data):
    st.subheader("Sparklines")

    # Generate random data points
    np.random.seed(42)
    num_rows, num_cols = data.shape
    plt.figure(figsize=(15, 5))

    # Plot sparklines for each row
    for i in range(num_rows):
        plt.subplot(num_rows, 1, i+1)
        plt.plot(data.iloc[i], marker='o', markersize=3, color='blue')
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.title(f"Row {i+1} Trend")
        plt.tight_layout()

    st.pyplot()

def fish_eye_distortion():
    # Generate random data points
    np.random.seed(42)
    x = np.random.randn(1000)
    y = np.random.randn(1000)

    # Apply fish-eye distortion based on user input
    distortion_level = st.slider("Distortion Level", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    x_distorted = r * np.cos(theta)**distortion_level
    y_distorted = r * np.sin(theta)**distortion_level

    # Plot original and distorted points
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].scatter(x, y, alpha=0.5)
    ax[0].set_title('Original Points')
    ax[1].scatter(x_distorted, y_distorted, alpha=0.5)
    ax[1].set_title('Distorted Points')
    plt.tight_layout()
    st.pyplot(fig)


import altair as alt

import seaborn as sns

def plot_pca():
    st.title("Multivariate Analysis")
    st.sidebar.title("Navigation")
    st.sidebar.write("Go to Text Analysis for sentiment analysis")
    st.write("Principal Component Analysis (PCA)")

    # Drop any non-numeric columns and columns with missing values
    data_numeric = data.select_dtypes(include=['number']).dropna()

    # Create a sidebar widget for selecting the plot type
    plot_type = st.sidebar.radio("Select Plot Type", ["Scatter Plot", "Biplot", "Variable Loadings", "Correlation Matrix"])

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_numeric)

    # Perform PCA
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(data_scaled)

    # Plot selected PCA result
    if plot_type == "Scatter Plot":
        # Convert PCA result to DataFrame
        pca_df = pd.DataFrame(pca_result, columns=['PCA Component 1', 'PCA Component 2'])

        # Scatter plot with brushing
        brush = alt.selection_interval()

        scatter_plot = alt.Chart(pca_df).mark_circle().encode(
            x='PCA Component 1',
            y='PCA Component 2',
            color=alt.condition(
                brush,
                alt.value('red'),  # Color for selected points
                alt.value('steelblue')  # Default color
            ),
            tooltip=['PCA Component 1', 'PCA Component 2']
        ).add_selection(
            brush
        ).properties(
            width=600,
            height=400
        )

        st.altair_chart(scatter_plot, use_container_width=True)
        
    elif plot_type == "Biplot":
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
        for i, (pc1, pc2) in enumerate(zip(pca.components_[0], pca.components_[1])):
            plt.arrow(0, 0, pc1, pc2, color='r', alpha=0.5)
            plt.text(pc1 * 1.15, pc2 * 1.15, data_numeric.columns[i], color='g', ha='center', va='center')
        plt.title('Biplot')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        st.pyplot()

    elif plot_type == "Variable Loadings":
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        plt.barh(data_numeric.columns, loadings[:, 0], color='b', alpha=0.5)
        plt.barh(data_numeric.columns, loadings[:, 1], color='r', alpha=0.5)
        plt.xlabel('Loadings')
        plt.title('Variable Loadings Plot')
        st.pyplot()
    
    elif plot_type == "Correlation Matrix":
        # Calculate correlation matrix
        corr_matrix = data_numeric.corr()

        # Plot correlation matrix heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        st.pyplot()


def text_analysis():
    st.title("Text Analysis")
    st.markdown("### Roast Co-occurrence Network")
    roast_cooccurrence_network()
    st.markdown("### Temporal Analysis")
    temporal_analysis()
    st.markdown("### Name Network")
    name_network()
    st.markdown("### Word Cloud")
    generate_word_clouds()
    st.markdown("### Sentiment Analysis")
    sentiment_analysis()
    



def generate_tree_data(df):
    # Group data by roaster and roast
    grouped_data = df.groupby(['roast', 'roaster']).size().reset_index(name='count')

    # Convert the grouped data to a hierarchical structure
    tree_data = {}
    for idx, row in grouped_data.iterrows():
        if row['roaster'] not in tree_data:
            tree_data[row['roaster']] = {}
        if row['roast'] not in tree_data[row['roaster']]:
            tree_data[row['roaster']][row['roast']] = row['count']
        else:
            tree_data[row['roaster']][row['roast']] += row['count']

    return tree_data

def roast_cooccurrence_network():
    st.subheader("Roast Co-occurrence Network")

    # Filter out rows with missing values in relevant columns
    data_filtered = data[['rating', 'roast']].dropna()

    # Group by roaster and create edges between roasts within the same group
    edges = data_filtered.groupby('rating')['roast'].apply(lambda x: list(zip(x[:-1], x[1:]))).tolist()

    # Create a network graph
    G = nx.Graph()

    # Add edges to the graph
    for edge_list in edges:
        G.add_edges_from(edge_list)

    # Compute node positions using Kamada-Kawai layout
    pos = nx.kamada_kawai_layout(G)

    # Draw the network graph
    plt.figure(figsize=(15, 10))
    nx.draw(G, pos=pos, with_labels=True, font_size=12, node_size=1500, node_color='red', edge_color='green', linewidths=0.5)
    plt.title("Roast Co-occurrence Network")
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)

import plotly.graph_objects as go

import plotly.graph_objects as go

def name_network():
    st.subheader("Name Network")

    # Filter out rows with missing values in relevant columns
    data_filtered = data[['roast', 'name']].dropna()

    # Group by roast and name
    roast_name = data_filtered.groupby(['roast', 'name']).size().reset_index(name='count')

    # Create a network graph
    G = nx.Graph()

    # Add nodes for names
    names = set(roast_name['name'])
    G.add_nodes_from(names)

    # Add edges between names based on shared roasts
    for roast, name_df in roast_name.groupby('roast'):
        names_in_roast = name_df['name'].tolist()
        for i in range(len(names_in_roast)):
            for j in range(i + 1, len(names_in_roast)):
                name_1 = names_in_roast[i]
                name_2 = names_in_roast[j]
                if not G.has_edge(name_1, name_2):
                    G.add_edge(name_1, name_2, weight=1)
                else:
                    G[name_1][name_2]['weight'] += 1

    # Create layout for visualization
    pos = nx.spring_layout(G, k=1)  # Use spring layout for better visualization

    # Get node positions
    node_x = []
    node_y = []
    for node, position in pos.items():
        node_x.append(position[0])
        node_y.append(position[1])

    # Create node trace
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            size=20,
            color='lightgreen',
            line_width=2
        ),
        text=list(G.nodes),
        showlegend=False  # Hide legend
    )

    # Create edge trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color='lightgray'),
        hoverinfo='none',
        mode='lines',
        showlegend=False  # Hide legend
    )

    # Create Plotly figure
    fig = go.Figure(data=[edge_trace, node_trace])

    # Update layout
    fig.update_layout(
        title='Name Network Graph',
        titlefont_size=16,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    # Show Plotly figure
    st.plotly_chart(fig, use_container_width=True)


def sentiment_description():
    st.subheader("Sentiment Description of desc_1")

    # Filter out rows with missing values in desc_1 column
    desc_1_data = data['desc_1'].dropna()

    # Calculate overall sentiment polarity of desc_1
    overall_sentiment = desc_1_data.apply(lambda x: TextBlob(x).sentiment.polarity).mean()

    # Define sentiment descriptions
    if overall_sentiment > 0:
        sentiment_desc = "The overall sentiment of the desc_1 is positive."
    elif overall_sentiment < 0:
        sentiment_desc = "The overall sentiment of the desc_1 is negative."
    else:
        sentiment_desc = "The overall sentiment of the desc_1 is neutral."

    st.write(sentiment_desc)


import plotly.graph_objects as go

def sentiment_analysis():
    # Assuming 'desc_1' contains the text data for sentiment analysis
    desc_1_data = data['desc_1'].dropna()

    # Perform sentiment analysis on desc_1 column
    sentiment_scores = desc_1_data.apply(lambda x: TextBlob(x).sentiment.polarity)

    # Count the number of positive, negative, and neutral sentiments
    num_positive = (sentiment_scores > 0).sum()
    num_negative = (sentiment_scores < 0).sum()

    # Calculate percentages
    total_reviews = len(sentiment_scores)
    percentage_positive = (num_positive / total_reviews) * 100
    percentage_negative = (num_negative / total_reviews) * 100

    # Create Plotly pie chart
    fig = go.Figure(data=[go.Pie(labels=['Positive', 'Negative'],
                                 values=[percentage_positive, percentage_negative],
                                 hole=0.5,
                                 marker_colors=['green', 'red'],
                                 textinfo='percent+label'
                                 )])

    fig.update_layout(title='Sentiment Analysis',
                      titlefont_size=24,
                      margin=dict(l=0, r=0, t=50, b=0))

    # Show Plotly figure
    st.plotly_chart(fig)

from wordcloud import STOPWORDS

def generate_word_clouds():
    st.subheader("Word Clouds for desc_1")

    # Filter out rows with missing values in desc_1 column
    desc_1_data = data['desc_1'].dropna()

    # Perform sentiment analysis on desc_1 column
    # Concatenate all desc_1 strings into a single string
    desc_1_text = ' '.join(desc_1_data)

    # Create sidebar widgets for customizing word cloud appearance
    background_color = st.sidebar.color_picker("Background Color", "#FFFFFF")
    max_words = st.sidebar.number_input("Max Words for word cloud", min_value=50, max_value=1000, value=200)
    stopwords = st.sidebar.text_area("Stopwords (separated by commas)", ", ".join(STOPWORDS))

    # Split the stopwords by commas and remove leading/trailing whitespaces
    stopwords = [word.strip() for word in stopwords.split(',')]

    # Generate word cloud with customized appearance
    wordcloud = WordCloud(width=800, height=400, background_color=background_color, colormap='inferno',
                          max_words=max_words, stopwords=stopwords).generate(desc_1_text)

    # Plot word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud')
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)

def linear_model(x, m, c):
    return m * x + c

def exponential_func(x, a, b):
    return a * np.exp(b * x)

def predict_growth_type(roast_temporal):
    growth_types = {}

    for roast in roast_temporal['roast'].unique():
        roast_data = roast_temporal[roast_temporal['roast'] == roast]
        x = np.arange(len(roast_data))
        y = roast_data['count'].values

        # Fit linear regression model
        lin_reg = LinearRegression().fit(x.reshape(-1, 1), y)
        linear_score = lin_reg.score(x.reshape(-1, 1), y)

        # Fit exponential growth model
        try:
            popt, pcov = curve_fit(exponential_func, x, y)
            exponential_score = np.mean((exponential_func(x, *popt) - y) ** 2)
            if exponential_score < linear_score:
                growth_types[roast] = "Exponential"
            else:
                growth_types[roast] = "Linear"
        except Exception as e:
            growth_types[roast] = "Linear"

    return growth_types

def temporal_analysis():
    st.subheader("Temporal Analysis")

    # Filter out rows with missing values in relevant columns
    data_filtered = data[['roast', 'review_date']].dropna()

    # Convert the review_date to datetime format
    data_filtered['review_date'] = pd.to_datetime(data_filtered['review_date'], format='%b-%y')

    # Group by roast and review date
    roast_temporal = data_filtered.groupby(['roast', pd.Grouper(key='review_date', freq='M')]).size().reset_index(name='count')

    # Predict growth types
    growth_types = predict_growth_type(roast_temporal)

    # Create a sidebar widget for selecting the roast
    selected_roast = st.sidebar.selectbox("Select Roast for Temporal Analysis", sorted(data_filtered['roast'].unique()))

    # Plot temporal analysis for the selected roast
    plt.figure(figsize=(12, 6))
    roast_data = roast_temporal[roast_temporal['roast'] == selected_roast]
    plt.plot(roast_data['review_date'], roast_data['count'], label=selected_roast, color='skyblue')
    plt.title(f"Temporal Analysis for {selected_roast} Roast")
    plt.xlabel("Date")
    plt.ylabel("Number of Reviews")
    plt.legend()
    st.pyplot()

    # Display growth type predictions with modified text size and color
    st.subheader("Predicted Growth Type")
    if growth_types[selected_roast] == "Exponential":
        st.write(f"<p style='font-size:20px; color:green;'>Growth Type: {growth_types[selected_roast]}</p>", unsafe_allow_html=True)
    else:
        st.write(f"<p style='font-size:20px; color:red;'>Growth Type: {growth_types[selected_roast]}</p>", unsafe_allow_html=True)






def main():
    set_custom_css()  # Set custom CSS styles
    
    # Initialize session state variables
    if 'page' not in st.session_state:
        st.session_state.page = "Table_Analysis"

    handle_page_navigation()

    # Listen for changes in the URL hash
    hash_code = st.session_state.get('hash_code', '')
    current_page = st.session_state.page

    if hash_code != '':
        js_code = f"""
        if (window.location.hash != "{hash_code}") {{
            window.location.hash = "{hash_code}";
            navigate("{current_page}");
        }}
        """
        st.write(js_code, unsafe_allow_html=True)

    # Handle page navigation based on the URL hash
    if current_page == 'Table_Analysis':
        explore_data()
    elif current_page == 'Multivariate_Analysis':
        plot_pca()
    elif current_page == 'Text_Analysis':
        text_analysis()

    top_navigation_bar()  # Add top navigation bar after rendering text analysis


if __name__ == "__main__":
    main()


   

   
