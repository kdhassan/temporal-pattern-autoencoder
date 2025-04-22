# AE-LSTM Policy Embedding Generator

This project introduces an LSTM-based autoencoder for capturing sequential behavior in datasets where each entity evolves over time, such as insurance policy records or customer journey logs. It is designed to transform policy-level sequences into compact vector embeddings and use those to measure similarity between different policy trajectories.

## 🧠 Components
- **SequenceGenerator**: Converts raw tabular data into structured, padded sequences suitable for model input.
- **GeneralLSTMAutoencoder**: Builds and trains an LSTM autoencoder. Also extracts compressed sequence representations and performs similarity search using FAISS.

## 🔄 Core Workflow
1. **Data Sequencing**: Convert tabular data into padded sequences grouped by unique IDs.
2. **Model Training**: Train an LSTM autoencoder on sequential inputs.
3. **Embedding Extraction**: Use the encoder to obtain fixed-length embeddings from sequences.
4. **Similarity Matching**: Identify closest sequences based on embedding proximity.

## 🧪 Code Sample
```python
from src.sequence_generator import SequenceGenerator
from src.ae_lstm import GeneralLSTMAutoencoder

# Convert raw DataFrame to sequence data
generator = SequenceGenerator(df=dummy_df, id_column='policynumber', sequence_column='pol_exp_year', target_column='acpt_nt_ind')
sequences, labels = generator.generate_sequences()

# Train the autoencoder
autoencoder = GeneralLSTMAutoencoder(sequences.shape[1], sequences.shape[2])
autoencoder.fit(sequences, epochs=50, batch_size=64)

# Generate compressed embeddings
embeddings = autoencoder.encode(sequences)

# Identify nearest sequences
distances, indices = autoencoder.find_nearest_neighbors(embeddings, embeddings, k=8)
```

## 📊 Sample Input Format
Each record should represent a point in a policy’s history:

| policynumber | pol_exp_year | feature_0 | feature_1 | ... | feature_9 | acpt_nt_ind |
|--------------|--------------|-----------|-----------|-----|-----------|--------------|
| 1            | 0            | 0.37      | 0.95      | ... | 0.71      | 1            |
| 1            | 1            | 0.02      | 0.96      | ... | 0.21      | 1            |
| ...          | ...          | ...       | ...       | ... | ...       | ...          |

## 🔍 Use Cases

- **Behavioral Matching**: Match current customers to past customers with similar progression for retention insights.
- **Risk Forecasting**: Identify sequences that resemble those of past high-risk clients or claims.
- **Churn Analysis**: Trace sequential pathways of churned accounts to preemptively flag at-risk policies.
- **Anomaly Detection**: Use embedding distance to detect outlier sequences.
- **Lifecycle Benchmarking**: Compare policy progression across cohorts for insight into product design or engagement.

## 🚫 Limitations of Traditional Models
Conventional ML models such as GLMs or gradient boosted trees often assume independence across rows, failing to account for sequence dynamics inherent in longitudinal datasets. This project overcomes those limitations by:

- Capturing temporal structure directly without manual lag features
- Handling irregular and variable-length sequences
- Embedding entire sequence patterns into compact representations
- Learning from end-to-end sequential data without extensive feature engineering

## ✅ Benefits
- Minimal preprocessing required
- Seamless integration with downstream clustering, classification, or recommendation systems
- Adaptable to both classification and unsupervised similarity tasks

## 🔧 Requirements
- Python 3.7+
- TensorFlow / Keras
- pandas, numpy, faiss-cpu

---

For more advanced applications or integration support, feel free to reach out!
