# Data Science

## Featured

### [Covid-19 Visualization Dashboard](Covid-Test)
An interactive Covid-19 data exploration project built on the 2020 Johns Hopkins dataset. Features an animated choropleth world map, bubble map visualization, mortality/recovery rates, WHO region comparison, worldometer analysis, USA county drill-down, and Holt-Winters forecasting.

### [Ebola Data Explorer](ebola)
An interactive Streamlit dashboard pairing visual exploration of historical Ebola outbreaks with small forecasting and modeling demos. Features six interactive charts (outbreak timeline, CFR vs. outbreak size, cumulative deaths, an Africa bubble map, a symptom co-occurrence heatmap, and a transmission risk lollipop) plus four method demos: a patient outcome classifier, CFR trend lines, a monthly case forecast, and an outbreak severity regression. Built with pandas, NumPy, Plotly, scikit-learn, and statsmodels. Datasets are deliberately tiny, so the models demonstrate method rather than operational prediction.

### [Protein Visualization](protein_visualization)
A Dash web app for visualizing 3D protein structures from user uploaded .pdb files. Includes RCSB-fetch-by-ID, chain/residue selectors, sequence view, stats panel, HETATM toggle, and mmCIF support.

### [Intrusion Detection - Computer Security](IntrusionDetection)
A predictive model used to classify network connections as benignor malicious across attack categories (DOS/DDOS, R2L, U2L, probing) using the KDD Cup dataset with pandas, NumPy, and Scikit-Learn. After preprocessing, 6 classifiers were compared (Naive Bayes, Decision Tree, Random Forest, SVM, Logistic Regression, Gradient Boosting), resulting in Random Forest performing the best with 99.88% prediction accuracy.

## Introductory

### [Exploratory Data Analysis](EDA)

A basic exploration of data analytics using python, seaborn, matplotlib, and pyplot. Uses the beer & brewery datasets to identify statistical values like standard deviation and correlations.

### [Iris Classification](Iris_Classification)
An exploration of the Iris Flower dataset, evaluation of various machine learning models in classification accuracy, and predictions using the most accurate model.

### [Linear Regression](Custom_LinearRegression)

Basic implementations of simple linear regression and multiple linear regression using the Boston Housing dataset.

### [Titanic Survival Prediction](Titanic)

Uses a random forest classifier to predict which passengers survive the Titanic. Achieved 83.8% accuracy.

## Classification & Prediction

### [Content Based Movie Recommendation System](MovieRecommendations)

Using a dataset of movies and associated credits, personalize movie recommendations using plot details and movie metadata. Created with Python, ast, sklearn, pandas, and numpy.

### [Heart Disease Prediction Model](HeartDiseasePrediction)
Developed and evaluated accuracy of various machine learning models in predicting heart disease development. Achieved 85.07% accuracy with Linear Regression.

### [Cancer Cell Classification](CancerCellClassification)
A scikit-learn script that trains a Gaussian Naive Bayes classifier on the built-in breast cancer dataset to predict if a tumor is malignant or benign. Achieved 94.15% accuracy with a 67/33 train/test split.

## Bioinformatics
### [Gene Expression](analyze_gene_expression)
A CLI for analyzing gene expression microarray data from the NCBI Gene Expression Omnibus (GEO). Auto-downloads datasets by accession via GEOparse, then runs EDA (PCA, sample boxplots, correlation heatmaps, gene-variance histograms), differential expression (Welch's t-test with Benjamini–Hochberg FDR), volcano plots, clustered heatmaps of top DE genes, and cross-dataset comparison (shared DE genes and log2 fold-change concordance). Built with pandas, NumPy, SciPy, statsmodels, scikit-learn, matplotlib, and seaborn; validated on the GSE19804 and GSE10072 lung cancer series.
