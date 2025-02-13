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

### Why Generative AI?

* Generative AI is transforming industries by creating new and original content.
* it powers applications such as text generation, image creation, and more.
* Business leverage generative AI to improve efficiency, personalization, and creativy.
* It is a type of AI focused on creating new, original content.
* AI generally predicts outcomes based on input data, while Generative AI focuses on producing new outputs that resemble the data it was trained on
* Transformer Network: A neural network architecture that processes input in parallel
* Context Window: The model's kind of memory span when it's generating text / Portion of input data that a model processes at a time
* Tokens: Smallest units of data (e.g words or parts of words) 
* Tokenization: The process of breaking down input into tokens
* Embeddings and vectors: Numeric representation of words or phrases that capture their meaning, used to understand the relationships between tokens (words) 
* Chunking: Used to handle large amounts of data by breaking it down into smaller, more manageable pieces called "chunks"
* LLMs (Large Language Models): Generative AI models trained on vast models of text data e.g ChatGPT
* Prompt Engineering: The input or instruction given to the model
  - techniques: 
    - Zero shot Learning: The AI recognizes something entirely new without any direct examples using descriptions or related knowledge.
    - One-shot Learning: The Ai learns from just one example and can recognize  it in the future.
    - Few-Shot Learning: the AI learns from a few examples and can generalize to recognize new instances.
* Multimodal Models: Handle different types of data (e.g text, images, audio)
* Diffusion Models: Mainly visual in nature, are used for generating high-quality images, audios, and video

### Uses cases and applications

* Text generation
* Summarization of long documents
* Code generation and completion
* 3D content creation
* Extract Key information from unstructured data

#### Architectures behind AI:

* Generative Adversial Network (GAN): Great for high quality images generation
* Variational Autoencoder (VAE): Great for unsupervised models
* Transformers

### Foundation Model Lifecycle

* Structured AI project lifecycles ensure smooth transitions from concept to deployment, optimizing scalability
1. Identify (Narrow Scope vs Broad Scope, Pre-Trained Models or Training From Scratch)
2. Experiment (try different models, evaluate performance, select most suitable model)
3. Adapt (Refine, prompt engineering, RLHF)
4. Evaluate (evaluate with real world conditions)
5. Deploy 
6. Monitor
* The role of prompt egineering:
  - Guide the model with clear input for accurate results.
  - Use examples to improve performance without extra training
  - Enhance the model with supervised learning if needed
* Reinforcement Learning From Human Feedback (RLHF): 
  - Align model with human preferences
  - Human Feedback fine-tunes model
  - Ethical/Subjective considerations
  - Minimizes toxic language and hallucinations
  - Makes models more helpful, honest, and harmless

### Generative AI Applications  - Capabilites and Limitations

* Challenges:
  - Limited Task Performance
  - Ethical Considerations
  - Risks in Sensitive Areas (Healthcare, Finance)
  - Organizational Commitment (Prioritize ethics to ensure societal benefits)
* Tracking Business Metrics With AI:
  - Key Metrics: (Accuracy, Eficiency, Conversion Rate)
  - Purpose of metrics: Asses AI value, Insights for Optimization
  - Desired Outcome: Improve ROI, Align with business objectives

### AWS Infrastructure for Building Gen AI Applications

* Amazon SageMaker
* Amazon Bedrock [Playground for Generative AI with Bedrock](https://partyrock.aws/)
* Amazon Q


## Applications of Foundational Models

### Design Considerations:

* Cost Considerations:
  - Simple Model: Acceptable accuracy lost --> Faster inference
  - Complex Model: High accuracy --> Slow inference
* Modality Considerations (types of input data a model can process): Text, Audio, Image 
* Choose the Right Architecture: 
  - Convolutional Neural Network CNNs (better for image recognition tasks)
  - Recurrent Neural Network RNNs (better for natural language processing tasks)
* Performance Metrics:
  - Accuracy: Overall correctness → (True Positives + True Negatives) / Total Predictions.
  - Precision: Correct positive predictions → True Positives / (True Positives + False Positives).
  - Recall: Ability to find all positives → True Positives / (True Positives + False Negatives).
  - F1 Score: Balance between precision & recall → 2 × (Precision × Recall) / (Precision + Recall).
* Model Customization:
  - Fine-tuning
  - Pre-Training

### Selecting Pre-Trained Models

* Mitigating Bias
* Availability and Compability (Model repository, framework)
* Model Maintenance and Updates
* Customization (does it accept fine tunning? or just re-training)
* Transparency (does the model show what it is doing?)
  - Interpretability: it is about understanding how a model works internally.
  - Explainability: it is about providing insights into why a model made a certain decision, even if the internal workings remain complex.
* Hardware Constraints
* Data Privacy Considerations
  - Federated Learning: Enables AI models to learn from decentralized data sources without compromising privacy.
* Transfer Learning

### Inference Parameters

* Parameters to control over model behavior and output characteristics
* Common Inference Parameters:
  - Temperature (Deterministic <> Creative)
  - Top-K (Limits selection to a fixed number of tokens)
  - Top-P (Dynamically adjusts the number of tokens considered based on probability distribution)
  - Length (number of tokens (words) generated for the model)
* Penalties and Stop Sequences:
  - penalties: Stop repeating phrases
  - Sequences: Stop sequences, useful for bullet lists

### Retrieval Augmented Generation (RAG)

* Enhances language models with external data, key for improving accuracy and relevance in AI tasks
* How RAG Works:
  - Retrieval: The system searches for relevant documents or information from a knowledge base (e.g., Wikipedia, private documents, APIs).
  - Augmentation: The retrieved data is fed into a generative AI model as additional context.
  - Generation: The AI generates a response based on both the retrieved documents and its own knowledge.
* Challenges and Limitations of RAG:
  - Require robus infrastructure
  - Data privacy and security corncerns

### Vector Databases on AWS

* Store data as embeddings wich are numerical representations of data like text and images
* OpenSearch for Generative AI
* Amazon Aurora PostgreSQL support pgvector
* Amazon Neptune ML (Graph Neural Network)
* Vector Search for Amazon MemoryDB
* Amazon DocumentDB (with MongoDB Compability)

### Foundational Models Customization Approaches 

* Pre-Training: High Flexibility, General-purpose learning from unstructured data.
* Fine-Tunning: Good Flexibility, critical to adapting a general-purpose-model to your **specific use cases**
  - Avoid Catastrophic forgetting in Fine-Tunning
    - PEFT Techniques (Parameter-Efficient Fine-Tunning):
      - LoRA (Low-Rank Adaptation): Freezes original weights except for low-rank weights 
      - ReFT (Representation Fine-Tunning): Modified model representations rather than weights
    - Multitask Fine-Tunning: Models are trained on multiple tasks simultaneously
    - Domain-Specific Fine-Tunning: is not a task specific tunning, helps to specialized models in fields (domains)
* In-Context Learning: Limited Customization
* RAG: Enhanced output with external data.
* Continous Pre-Training: Essential for keeping foundational models current, relevant, and performing well in evolving environments

### Prompt Engineering 

* Prompt ia an input provided by the user to guide an LLM.
* Componentes of a prompt:
  - Task
  - Context
  - Input Text (content, examples)
* Negative Prompts: Guides the model to avoid certain responses or behaviors
* Model's Latent Space: Representation of the data that a model has

``` 
Prompt Example:

Task: Generate a fun and informative description of a Siberian Husky's personality.

Context: Siberian Huskies are energetic, intelligent, and independent dogs known for their playful nature and strong-willed temperament. The description should highlight their friendly and mischievous side while keeping it engaging.

Input Text: Woz, my Husky, loves running in the snow, howling at sirens, and outsmarting me when it's time for a bath. He’s both a lovable goofball and a master escape artist.

Negative Prompt: Avoid mentioning aggression, excessive barking, or portraying the Husky as a dangerous or overly obedient dog. Keep the tone light and fun. 
```

* Zero-Shot Prompting: Prompt without example
* One-Shot Prompting: Prompt with one example
* Few-Shot Prompting: Prompt with numerous examples
* Common Risks in Prompt Engineering (Amazon bedrock guardrails set limits for this):
  - Exposure: Leakage of sensitive or private data due to poorly crafted prompts.
  - Poisoning: Manipulation of model outputs by injecting biased or harmful data.
  - Hijacking: Unauthorized control over the prompt to alter intended responses.
  - Jailbreaking: Bypassing model restrictions to generate prohibited content.

### Evaluating Foundational AI Models

* Recall-Oriented Understudy for Gisting Evaluation (ROUGE):  Measures text summarization quality by comparing overlap with a reference.
* Bilingual Evaluation Understudy (BLEU): Evaluates machine translation accuracy using n-gram overlap.
* LLM Benchmarking: Assessing large language models on various tasks for performance comparison.
* General Language Understanding Evaluation (GLUE): A benchmark for evaluating general NLP understanding across multiple tasks.
* SuperGLUE: An improved version of GLUE with more challenging language tasks.
* Massive Multitask Language Understanding (MMLUE): Tests language models on a diverse set of real-world, multitask evaluations.
* Big Bench:  A large-scale benchmark covering diverse reasoning and knowledge tasks.
* Holistic Evaluation of Language Models (HELM): A holistic framework for evaluating language models across fairness, bias, and performance.
* SageMaker Clarify: Manual Evaluation
* Amazon Bedrock - BERTScore: Provides an evaluation module that automatically compares generated responses

## Guidelines for Responsible AI

### Responsible AI - Features

* Core Dimentions:
  - Fariness: Equal treatment for users
  - Explainability: Importance of explaining AI decisiones (e.g. loan rejection)
  - Robustness: Stay accurate and reliable despite unexpected, noisy, or adversarial inputs
  - Privacy and Security: Protecting user data
  - Gorvernance: Legal compliance requeriments

### Tools for Identifying Responsible AI Features

* SageMaker Clarify: Identify bias during different stages of the ML project also enhance explainability, some features are:
  - Bias detection during data preparation (aspects to analyze: Data balance and group representation)
  - Bias detction after model training
  - Explaining model decisions (transparency and explainability)
* Amazon Bedrock for Guardrails: Security service that can limit or minimize outputs from a model
  - Ensure your AI applications are safe and ethical

### Legal Risks in Generative AI

* Hallucination: when the model lies but is sounds really accurate
* Legal Challenges, copyright
* Bias: is a major issue, particulary in decision-making processes (e.g rejecting people in a hiring process because their age or gender)
* Offensive and Innapropiate AI outputs: many times is because it was trained on inappropiate data (Guardrials is so appropiate for this).
* Data Privacy and Security Risks: Sensitive information may appear in model outputs unintentionally

### Dataset Characteristics and Bias

* Balanced dataset matter in AI.
  - Inclusive and Diverse Data Collection
  - Data Curation (Labeling, Cleaning, Augmenting, Balancing)
  - Data Processing Techniques (Cleaning, Normalization, Feature Selection)
  - Data Augmentation for Balancing Datasets
  - Regular Auditing for Fairness
* SageMaker Clarify: 
  - Helps identify and mitigate bias in data
  - offer tools to explain models predictions
  - Automates fairness and transparency checks
* SageMaker Data Wrangler: 
  - Helps to identify unbalanced datasets
  - Provides tools for data cleaning and augmentation

### Transparent and Explainable Models

* Key Aspects of Transparency:
  - Interpretability: 
    - Simple models with clear rules
    - How easily we can understand the mechanics of the model.
    - Some industries require hogh level of interpretability (e.g finance, healthcare)
  - Explainability: 
    - Complex models viewed as black boxes
    - Describing why a model produced a certain output without knowing how it works internally
    - Neural Networks are not easily interpretable
* AWS AI Transparency - Service Cards: 
  - Provide to users with details about the model that they want to to use, available for recognition, textract, comprenhend 
  - SageMaker Model Cards: allow to provide clear information about the model,its training, metrics and more to promote transparency.
* Human-Centered-AI Prioritizing Human Needs:
  - Involving users in AI design and development 
  - Amazon Augmented AI (A2I): allow human review of AI predictions
  - Amazon SageMaker Ground Truth: Incorporate human beings into the human loop learning (RLHF)
* Amplified Decision Making: AI enhances human decision-making by providing insights, predictions, and automation.

## Security, Compliance, & Governance for AI solutions

### Securing AI Systems with AWS Services

* AIM: Policies and Permissions, MFA
* CloudTrail: For activity logging and auditing
* S3 block public access
* AWS Key Management Service (KMS)
* PrivateLink: Enables private connections between VPCs and AWS or third-party services.
  - VPC Endpoints → Allow VPCs to connect privately to AWS services, using either:
    - Interface endpoints (use PrivateLink)
    - Gateway endpoints (do not use PrivateLink, only for S3 & DynamoDB)
* SageMaker Distributed Training - Inter-Node Encryption

### Source Citation and Data Lineage

* Machine Learning - Need for Tracking Artifacts (Reproducibility, Compliance, Regulatory Requirements):
  - Source Code: Github
  - Datasets: Amazon S3
  - Container Images: ECR
  - Model Artifacts: SageMaker Model Registry for Model Versioning
* SageMaker Model Cards: Provide information about a model has its steps through lifecycle:
  - Intended Uses
  - Risk assessments 
  - Training Details
  - Evaluation Results
* SageMaker Lineage Tracking: Create a graphical lineage of the workflow of the model
* SageMaker Feature Store: Store features sets
  - It also has feature lineage
  - Data Cataloging 
  - Point-in-time queries

### Security and Privacy Considerations

* Gen AI vulnerabilities:
  - Data Poisoning: e.g altered images can bypass facial recognition models
* Model Inversion and Reverse Engineering Threats
* Prompt Injection Attacks 

### Regulatory Compliance Standards fo AI Systems

* Compliance standars safeguard: Business and Consumer
* ISO:
  - ISO 42001 - 2023: Managing Risks
  - ISO 23894 - 2023: Being responsible with AI
* EU AI act: Categorizes applications by risk
* NIST RMF: Voluntary Framework for trustworthy AI
* Algorithmic Accountability Act: Transparency in AI decisions

### AWS Services for Governance and Compliance

* AWS Artifact: Contains every single regulatory attestation and compliance that AWS has achieved globally
* AWS Glue DataBrew: Data preparation for governance
* AWS Lake Formation: Fine grained data access control
* Amazon SageMaker Clarify: analyze what is going on the model, detecting bias
* AWS Config: Configuration tracking
* Amazon Inspector: Scanner for machine, containers, serverless check know CVEs
* AWS Audit Manager: Collect information for auditing 
* AWS cloudtrail: Activity Logging
* AWS Trusted Advisor: Best Practices and Compliance Recommendations
* AI Data Governance Strategies:
  - Pilars:
    - Availability
    - Integrity
    - Security