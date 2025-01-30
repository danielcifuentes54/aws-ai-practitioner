<img src="icons/AWS-AI.png" />

# AWS AI Practitioner
Repository to present to the AWS Medellin group relevant information to ace the AWS Certified AI Practitioner certification

## Fundamentals of AI and ML

### Artificial Intelligence (AI)

AI is a branch of computer science focused on creating systems capable of performin tasks that typically require human intelligence.

--> General Intelligence Simulations

1. Visual Percepcion
2. Speech Recognition
3. Decision making
4. Language Translation

#### Narrow AI

AI desgined for specific tasks , like virtual assistants 

#### General AI

Theoretical AI with human-like reasoning across diverse domains


### Machine Learning (ML)

ML involves using algorithms and statiscal models to allow computers to perform tasks by learning from data rather than following explicit instructions

--> Learning from data patterns

### Deep Learning

Deep Learning is a subset of ML that uses multi-layered neural networks to model and solve more complex problems

--> Complex Patterns Recognition

### Inferencing 

Inferecing is the process in which AI models make predictions or decisions using new data, types:

* Real Time Inferecing (chatbots, fraud detection, autonomous driving systems)
  * SageMaker provide real time endpoints for scalable and low latency processing
* Batch Inferecing (Sentiment analysis on social media post collected over a day)
  * SageMaker transforms jobs for applying models to datasets in AmazonS3

### Data Types

Data Types refer to the various forms in which data can be represented and processed by AI models. These include numerical, categorical, and unstrcutured data

#### Categorical Data

Categorical data refers to information that can be divied into categories or groups

#### Types

* Numerical Data
* Raw Text Data (unstructured data)
* Image Data
* Audio Data (time series data)

#### Labeled vs Unlabeled

Labeled data contains both the input data and the corresponding output (target variable), while unlabeled data has only the input whithout the expected output

* Labeled Data (used it in supervised learning)
* unlabeled Data (used it in unsupervised learning)

#### Imbalanced Data 

Imbalanced data refers to datasets where on class (in classifications problems), has significantly more examples than the other(s). This can result in biased models that favor the dominant class.

* ROC-AUC: A performance metric to help to mitigate imbalanced data

#### Big Data

Still relevant to manage large and complex datasets, to use it then in training, some AWS that can help with big data:

* Amazon Elastic Map Reduce (EMR)
* Amazon Glue
* Amazon SageMaker

#### Data Type for AI Models

1. Decision Trees: Structured Data
2. CNNs: Unstructured Data
3. RNNs: Time-Series Data

### Supervised, Unserpervised and Reinforcement Learning

* Supervised: Models learns from labeled data, e.g: predictions stock market
* Unservised: Finding patterns in unlabeled data, e.g: Identifying hidden customer segments based on purchasing behavior
* Reinforcement Learning: Models learns trough try and error by receiving rewards or penalties e.g: AlphaGo, Chess

### AI Limitations

* High costs
* Lack of Interpretability

### ML Development Lifecycle

Machine learning models are dynamic and require continous updates and retraining 

0. Business Goal Identification
1. Data Collection
  - 1.1 Data Preprocessing
  - 1.2 Data Augmentation
  - AWS Services: AWS S3, AWS Glue
2. Training
  - 2.1 Hyperparameter Tuning
  - 2.2 Evaluating Model Performances (Accuracy, precision, Recall, F1 score)
  - AWS Services: Amazon SageMaker
3. Deployment (Real Time, Batch)
  - AWS Services: Amazon SageMaker, Lambda, Amazon EMR
4. Monitoring
  - AWS Services: Amazon SageMaker, CloudWatch

### MLOps Concepts

* Automate ML deployment, monitoring and updates
* Uses CI/CD for continouos model delivery
* For the creation of MLOps pipelines you can use
  * Amazon SageMaker
    - Amazon SageMaker Pipelines
    - Version Control
    - Automating Model Training and Deployment 
    - Monitoring and Retraining Models (Amazon SageMaker Model Monitor)
    - Compliance and Auditability
    - Improving Model Quality (Amazon SageMaker Clarify)
  * Apache Airflow
* Model Performance Metrics

| Metric                        | What It Minimizes                          | Ideal When...                                                        |
|-------------------------------|--------------------------------------------|----------------------------------------------------------------------|
| **Precision**                 | False Positives                           | Being overly cautious about positive predictions is acceptable.      |
| **Recall**                    | False Negatives                           | Missing true positives is highly undesirable.                        |
| **F1 Score**                  | Trade-off between FP/FN                   | A balance of precision and recall is required.                       |
| **Area Under the Curve (AUC)**| Misclassification across thresholds for binary classification | Evaluating the model's ability to distinguish between positive and negative classes across all decision thresholds. |
| **Mean Squared Error (MSE)**  | Squared differences between actual and predicted values | Continuous predictions where large errors are penalized more. |
| **Business Metrics**          | Business-specific goals (e.g., revenue, cost savings, churn rate) | Evaluating the model’s direct impact on business outcomes. |

### AI and ML Services on AWS

* AI/ML - Service Categories:
  - Pre-trained AI Services (Ready to Use)
  - ML Platforms (Tools for custom ML model development)
  - Infrastructure for custom AI (Training infrastructure, including GPU instances)

#### AWS Glue

* Fully managed extract, transform, and load (ETL) service (serverless ETL). It helps you prepare and integrate data from various sources for analytics, machine learning, and application development.
* Uses a crawler to extract data and put it into  a data catalog, then we can have our own job to normalize, transform and then push the data to another datasource (Athena, Redshift, EMR) 
* Glue DataBrew: No-code data preparation tool designed for business analysts, data analysts, and data scientists to clean and normalize data

#### Amazon EMR (Elastic Map Reduce)

* Managed platform that allows you to run big data frameworks (like Apache Hadoop and Apache Spark).
* It is used on large-scale data analytics and machine learning
* Full control over cluster size (there's also a serverless version)

#### Amazon Bedrock: 

* Fully managed service for foundation models that simplifies **Generative AI** application development.
* Access to diverse foundation models.
* Easy customization with your data
* Seamless integration with AWS services
* AWS Bedrock Guardrials: Limiter for bedrock to ensurethe model does not stray beyond you want it to do.
* AWS Bedrock Agents: Automate complex tasks, Orchestrate workflows based on AI outputs, Enhance application capabilities

#### Amazon SageMaker

* Managed service to quickly buildind, training and deploying ML models, simplifying developmnet
* Workflow:
  - Data Ingestion: Amazon S3 integration
  - Data preparation and Exploration: SageMaker Notebook and SageMaker Data Wrangler
  - Model Training: SageMaker Training Jobs
  - Model Avaluation and Tunning: Validation data and Automatic model tunning
  - Model Deployment: SageMaker Endpoints
* Features:
  - Built-in algorithms and BYOA
  - Integrated Jupyter Notebooks
  - Distributed Training Jobs
  - Automatic Model Tunning 
  - SageMaker Studio (In browser IDE for ML)

#### Amazon Recognition

* Visual Analysis (images and videos) for object detection, facial analysis, and text recognition
* Features:
  - Object and Scene detection
  - Facial Analysis and recognition
  - Text in image
  - Activity Detection
  - Unsafe content Detection
  - Celebrity Recognition
  - Custom Labels
  - Integration with other AWS Services
  - Emotion Detection
  - Real-time Analysis

#### Amazon Lex

* Enables building chatbots and virtual assistants using voice and text. It powers Amazon Alexa and supports AI-driven customer support
* Features:
  - Natural Language Understanding (NLU) and Automatic Speech Recognition (APR)
  - Easy to Build
  - Fully Managed
  - Built-in Integrations
  - Multi-channel support

#### Amazon Poly

* Voice-based system, to convert text to speech
* Features:
  - Lifelike speech
  - Real-time streaming or file generation
  - SSML support (control/configurate speech)
  - Integration with other services (e.g lex, Lambda, s3)

#### Amazon Transcribe

* Voice-based system, to convert speech to text

#### Amazon Comprehend

* Understanding sentiment, intention, key phrases, important terms in text.

#### Amazon Fraud Detector

* Fully managed service which detects fraudelent activity
* Categorization tool

#### Amazon translate

* Convert text from one language to another one

#### Amazon Textract

* OCR: Object Character Recognition, text extract from documents

#### Amazon Augmented AI (Amazon A2I)

* Allows you to integrate human review into machine learning (ML) workflows. It’s designed for use cases where AI-generated predictions need human validation

#### Amazon QuickSight

* Visualization tool with interactive dashboards that uses SPICE (superfast parallel in-memory calculation engine)

## Fundamentals of Generative AI

###

## Applications of Foundational Models

###

## Guidelines for Responsible AI

###

## Security, Compliance, & Governance for AI solutions

###

