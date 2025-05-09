#!/bin/bash

# Create necessary directories
mkdir -p reports/logs
mkdir -p reports/figures

# Function to extract timestamp from log file
extract_timestamp() {
    local log_file=$1
    # Get the last line containing "Training complete" and extract timestamp
    grep "Training complete" "$log_file" | tail -n 1 | grep -o '[0-9]\{8\}_[0-9]\{6\}'
}

# Run all training scripts and collect timestamps
echo "Starting training runs..."

# Run MSCRNN
echo "Training MSCRNN..."
python training/train_mscrnn.py > reports/logs/mscrnn_training.log
mscrnn_timestamp=$(extract_timestamp reports/logs/mscrnn_training.log)

# Run MSCNN
echo "Training MSCNN..."
python training/train_mscnn.py > reports/logs/mscnn_training.log
mscnn_timestamp=$(extract_timestamp reports/logs/mscnn_training.log)

# Run CNN
echo "Training CNN..."
python training/train_cnn.py > reports/logs/cnn_training.log
cnn_timestamp=$(extract_timestamp reports/logs/cnn_training.log)

# Run CRNN
echo "Training CRNN..."
python training/train_crnn.py > reports/logs/crnn_training.log
crnn_timestamp=$(extract_timestamp reports/logs/crnn_training.log)

# Save timestamps to a file
echo "Saving timestamps..."
cat > reports/logs/training_timestamps.txt << EOF
MSCRNN: $mscrnn_timestamp
MSCNN: $mscnn_timestamp
CNN: $cnn_timestamp
CRNN: $crnn_timestamp
EOF

# Run analysis for each model
# echo "Running analysis scripts..."

# echo "Analyzing MSCRNN results..."
# python scripts/analyze_metrics.py "$mscrnn_timestamp"

# echo "Analyzing MSCNN results..."
# python scripts/analyze_metrics.py "$mscnn_timestamp"

# echo "Analyzing CNN results..."
# python scripts/analyze_metrics.py "$cnn_timestamp"

echo "Analyzing CRNN results..."
python scripts/analyze_metrics.py "$crnn_timestamp"

# echo "All training and analysis complete!"
echo "Results can be found in:"
echo "- Training logs: reports/logs/"
# echo "- Analysis results: reports/figures/metrics_*/"
echo "- Timestamps: reports/logs/training_timestamps.txt" 