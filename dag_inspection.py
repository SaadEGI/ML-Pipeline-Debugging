import networkx as nx

# Create a directed graph
pipeline_graph = nx.DiGraph()

# Add nodes (these could be derived from parsing your code or from MLInspect)
pipeline_graph.add_node("read_csv", label="Data Ingestion")
pipeline_graph.add_node("preprocess", label="Preprocessing")
pipeline_graph.add_node("train_model", label="Model Training")
pipeline_graph.add_node("predict", label="Prediction")

# Add edges to represent flow
pipeline_graph.add_edge("read_csv", "preprocess")
pipeline_graph.add_edge("preprocess", "train_model")
pipeline_graph.add_edge("train_model", "predict")
