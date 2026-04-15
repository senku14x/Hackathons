# Email Engagement Prediction

This project was developed for the Pan IIT AI/ML Hackathon (IDFC Bank and IIT Guwahati), where our team placed among the top 6 out of 5000+ teams nationwide.

## Objective
Predict the optimal engagement time for promotional emails to maximize user interaction using a large-scale dataset of user behavior.

## Dataset
- 8.7 million rows
- 400+ columns
- Included user actions and timestamps

## Methodology
- **Feature Engineering**: Converted timestamps into 28 weekly 3-hour time slots
- **Model**: LightGBM
- **Validation**: 5-fold cross-validation, early stopping
- **Regularization**: L1 & L2
- **Handling Scale**: Chunk-based data loading and preprocessing

## Results
- **Metric**: Mean Average Precision (MAP)
- **Score**: 0.84
- Achieved strong generalization and stability

## Tech Stack
`Python`, `LightGBM`, `Pandas`, `NumPy`, `Matplotlib`, `Scikit-learn`

