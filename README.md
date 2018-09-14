# Home-Credit-Default-Risk-Kaggle

This repo contains an indepth exploratory data analysis (EDA) notebook and a pipeline to create features, train models and produce predictions in the context of the Home Credit Default Risk Kaggle.

Within the EDA we investigate the test and training set distributions and key differences among them with conclusions on how those differences might be dealt with. 

The pipeline serves as a vehicle for performing the data science task of this competition. In the pipeline we load and clean the data from pertinent conclusions from the EDA. We train the data using lightGBM a gradient boosting framework that uses tree based learning algorithms. We finally create a csv file to submit predictions to the leaderboard.
