# Project Overview

This project implements an end-to-end machine learning pipeline for an e-commerce platform. The main objectives are:

1. Develop a model to predict a customer's purchase amount for the next month.
2. Implement a model registry service for managing ML models.
3. Create an inference service for making predictions.
4. Design a system for generating monthly purchase amount forecasts.

The project is structured into several microservices:

- Model Registry Service: Manages model versions and metadata.
- Preprocessing Service: Handles data cleaning and feature engineering.
- Inference Service: Provides predictions based on the trained models.

Tech stack of the project:

- Python
- XGBoost
- Scikit-learn
- Pandas
- FastAPI
- SQLAlchemy
- Docker
- Kubernetes (for deployment and scaling)

The project follows best practices for machine learning engineering, including version control, containerization, and scalable microservices architecture.
