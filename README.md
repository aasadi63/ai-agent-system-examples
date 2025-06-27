# ai-agent-system-examples
This repository was created to store various AI agent practices.

## Example for testing AI-AGENT
this part is created for testing if AI-Agent can understand the content of the repo

### Applicant Eligibility Classifier
This repository contains a binary classification machine learning model that predicts whether an applicant is **eligible** or **not eligible** based on their basic information. The model is designed for use in application processing pipelines and automates initial screening decisions.

### Project Overview
- **Goal**: Predict applicant eligibility (`eligible` / `not eligible`)
- **Model Type**: Binary Classification
- **Status**: Prototype / Production-ready *(choose one)*
- **Tech Stack**: Python, scikit-learn, pandas, Jupyter

### Inputs
The model expects applicant data with the following fields:

| Field Name       | Type     | Description                                 |
|------------------|----------|---------------------------------------------|
| `age`            | Integer  | Age of the applicant                        |
| `education`      | String   | Highest level of education (e.g., "Bachelor", "High School") |
| `employment_years` | Float  | Total years of employment                   |
| `marital_status` | String   | Marital status (e.g., "Single", "Married")  |
| `income`         | Float    | Annual income in USD                        |
| `has_criminal_record` | Boolean | Whether the applicant has a criminal record (`true` / `false`) |

*You can customize this table based on your actual schema.*

### Outputs
The model outputs a prediction in the following format:

| Field Name      | Type     | Description                                 |
|------------------|----------|---------------------------------------------|
| `applicant_id`   | String   | Unique identifier for the applicant         |
| `is_eligible`    | Boolean  | `true` if eligible, `false` otherwise       |
| `confidence_score` | Float  | Model's confidence in the prediction (0.0 - 1.0) |
