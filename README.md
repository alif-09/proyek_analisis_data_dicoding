# E-Commerce Data Analysis Project

## Setup Environment - Anaconda

1. Create a new environment and activate it:

    ```bash
    conda create --name main-ds python=3.9
    conda activate main-ds
    ```

2. Install all required libraries:

    ```bash
    pip install -r requirements.txt
    ```

## Setup Environment - Shell/Terminal

1. Create a new folder for the project:

    ```bash
    mkdir data_analysis_project
    cd data_analysis_project
    ```

2. Install dependencies using `pipenv`:

    ```bash
    pipenv install
    pipenv shell
    ```

3. Install the required libraries:

    ```bash
    pip install -r requirements.txt
    ```

## Running the Streamlit Application

Once the environment is set up successfully, run the Streamlit dashboard application with the following command:

```bash
streamlit run dashboard.py
```
