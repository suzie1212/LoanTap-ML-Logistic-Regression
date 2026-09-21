# 💳 LoanTap Credit Underwriting Dashboard

An end-to-end interactive machine learning dashboard built to evaluate personal loan creditworthiness, identify default probability, and optimize the Precision-Recall tradeoff between Non-Performing Assets (NPAs) and interest revenue.

## Live Demo
🔗 **Live App Link:** https://7g6jgjpgr7mcr7dhkieaa4.streamlit.app/

## Key Features
- **Executive Portfolio View:** Interactive segmentation across credit grades, terms, and housing statuses.
- **Credit Risk Drivers:** Visual analysis of DTI ratios, interest rates, and loan durations against default outcomes.
- **Inference Engine:** Real-time risk probability calculation for prospective loan applicants.
- **Threshold Simulator:** Commercial economic tradeoff modeling (losses avoided vs. interest income foregone).

## Local Installation
```bash
git clone [https://github.com/](https://github.com/)<your-username>/loantap-underwriting-dashboard.git
cd loantap-underwriting-dashboard
pip install -r requirements.txt
streamlit run app.py
