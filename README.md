# Latent Structures in Dating Preferences

![Team 15 Capstone Banner](banner.png)

## 📌 Overview

The code within this repo is tied to Team 15's 2026 MADS Capstone. It explores the dynamics of matchmaking by using machine learning to identify latent structures in dating preferences. It contains our python files that were used for the EDA and analysis mentioned in the written report. It also includes an interactive dashboard. 

### **Machine Learning Techniques Utilized in Our Project**

  * Random Forest 
  * K-Means 
  * Logistic Regression 

### **Data Access Statement**
Our project used subsets of data derived from the sources below. 

**1. Columbia Speed Dating Dataset**
*  The raw dataset was originally compiled by Ray Fisman and Sheena Iyengar from Columbia University. It is publicly available for academic use on Kaggle.
* **Original Source:** [Speed Dating Experiment](https://www.kaggle.com/datasets/annavictoria/speed-dating-experiment)
* **Repository Access:** The preprocessed and binned version used for this project is available within this repository at `data/Columbia_filtered.csv`.

**2. Stanford How Couples Meet and Stay Together (HCMST)**
* The raw longitudinal survey data (`stanford-hCMST.tsv`) is maintained by Stanford University. It is avialable to use as long as you agree to not identify individuals, charge to distribute it to others, and cite their work.
* **Original Source:** [Stanford Data Portal](https://data.stanford.edu/hcmst)
* **Repository Access:** The specific subset of variables filtered for this analysis can be found at `data/hCMST_filtered.csv`. Data dictionaries defining the variables are available in the `docs/` folder.

**3. OKCupid Dataset**
* The raw OKCupid profile dataset was originally scraped and published with permission for educational use. It is publicly available for on Kaggle. 
* **Original Source:** [OkCupid Profiles](https://www.kaggle.com/datasets/andrewmvd/okcupid-profiles)
* **Repository Access:** The OKcupid dataset contains 60k rows and is 137.31MB in size. Due to GitHub file size limitations, it was not uploaded into this repository. The data dictionaries are provided in the `docs/` folder.


-----

## 🚀 Getting Started

### Prerequisites

This project relies on the following core libraries:

- [Dash](https://dash.plotly.com/)
- [Pandas](https://pandas.pydata.org/docs/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Plotly](https://plotly.com/python/)
- [Matplotlib](https://matplotlib.org/stable/)
- [Seaborn](https://seaborn.pydata.org/)
- [Numpy](https://numpy.org/doc/)
- [NLTK](https://www.nltk.org/)

-----
### Installation & Setup

**1.  Clone the Repo**

```bash
    git clone https://github.com/Rodea41/Capstone_Team15.git
    cd Capstone_Team15
```

**2.  (Optional) - Create and Activate a Virtual Environment**

```bash
    # Windows
    python -m venv .venv
    .venv\Scripts\activate

    # Mac/Linux
    python3 -m venv .venv
    source .venv/bin/activate
```

**3.  Install Dependencies**
    
```bash
    pip install -r requirements.txt
```

### Run Code | For Visuals

**4.  Generate Visual**

- Visuals for the supervised learning models:

```bash
    python Model/predictivemodelsupervisedlearning.py
```

- Visuals for the homophily vs. latent trait analysis:

```bash
    python Analysis/homophily_vs_latent_analysis.py
```

### Run Code | For Dashboard

**5.  Run Dashboard**

```bash
    python main.py
```

---

## 👥 Our Team

  * **Sophia Chang**
  * **Rohit Baddam** 
  * **Christopher Rodea**

-----

## 📜 License

This project is licensed under the **[MIT/Apache/GPL]** License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.



## 🤖 Acknowledgments

* The layout, formatting, and images used in this README were generated with the assistance of AI. The actual content was completed by members of Team 15
* All other AI assistance (if any) is clearly cited within the python or Jupyter notebooks where it is used. 
