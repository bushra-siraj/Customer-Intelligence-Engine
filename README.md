# 🛍️ ShopAnalyzer AI: Unsupervised Customer Insights

### **Project Overview**
This project applies advanced **Unsupervised Learning** techniques to a dataset of **1.06 Million** e-commerce transactions. The goal is to move beyond basic reporting and uncover hidden customer behaviors, identify high-risk anomalies, and build an automated recommendation engine.

---

### **🚀 Key Features & Task Breakdown**

#### **Task 1: Data Preprocessing**
* **Action:** Handled 240k+ missing Customer IDs and normalized features using `StandardScaler`.
* **Insight:** Scaling was crucial to ensure that high-value "Total Spend" didn't mathematically drown out "Purchase Frequency."

#### **Task 2 & 4: Segmentation & PCA**
* **Method:** K-Means Clustering + Principal Component Analysis.
* **Observation:** We identified **4 distinct personas**. Using PCA, we reduced the data to two dimensions while maintaining **96.01% of the original variance**.
* **Segments:** 💎 VIPs, 🔄 Loyalists, 🌱 Newcomers, and ⚠️ At-Risk shoppers.

#### **Task 3: Anomaly Detection**
* **Method:** Gaussian Density Estimation.
* **Result:** Flagged **60 customers** in the 1% lowest density zones. These represent potential wholesale buyers or irregular system entries requiring manual review.

#### **Task 5: Recommendation System**
* **Method:** User-Based Collaborative Filtering via Cosine Similarity.
* **Logic:** Predicts future purchases by identifying "Taste-Twins"—users with nearly identical shopping history.

---

### **📊 Analysis & Reflection (Task 6)**
* **Hidden Patterns:** Unsupervised learning revealed that a customer's value is the relationship between visit frequency and total monetary value.
* **Comparison:** * **Clustering** provides the **Strategy**.
    * **Anomalies** provide the **Security**.
    * **PCA** provides the **Clarity**.
    * **Collaborative Filtering** provides the **Revenue**.
* **Real-World Use:** This pipeline is applicable to Fraud Detection and personalized marketing engines like those used by Amazon and Netflix.

---

### **🛠️ Tech Stack**
* **Language:** Python 3.x
* **Libraries:** Pandas, Scikit-Learn, Matplotlib, Seaborn
* **Interface:** Streamlit (Web Dashboard)

---

### **👩 Author**
* **Name:** **Bushra Siraj**
* **Email:** **[BushraSiraj586@gmail.com]**
* **Linkedin:** **[www.linkedin.com/in/bushrasiraj]**
