# Professional Certificate in Machine Learning and Artificial Intelligence Program from UC Berkeley Executive Education

# 20.1

# INTRODUCTION
This repository contains my capstone project for the Professional Certificate in Machine Learning and Artificial Intelligence Program from UC Berkeley Executive Education. The purpose of my project is to leverage the skills acquired throughout the course to build a trained model that can be used to analyze prescription claim data in near-real time. The trained model will use previously identified fraudulent claims for a single quarter then will be applied to the remaining 3 quarters to predict fraud with at least a 70% threshold.

# DATASET
I have obtained 2023 claim history for a single health plan. The data has been cleaned up and any PHI/PII has either been masked or removed. The following query was used to pull the dataset.

```
select distinct
md5(rx.claim_reference_number)::varchar(10) as masked_claim_reference_number,
rx.date_filled,
claim_type,
rx.customer_id,
rx.patient_paid_amount,
rx.copay_amount,
rx.basis_of_reimbursement,
rx.refund,
rx.transaction_status,
md5(rx.cardholder_id)::varchar(10) as masked_cardholder_id,
case when date_of_birth::text ~ '^\d{8}$'
    then extract(year from age(to_date(date_of_birth::text, 'YYYYMMDD')))
    else null
    end as age,
rx.prescriber_id,
presc.provider_name as prescriber_name,
ps.specialty_name,
presc.physical_city as prescriber_city,
presc.physical_state as prescriber_state,
rx.pharmacy_npi,
pharm.provider_name as pharmacy_name,
pharm.physical_city,
pharm.physical_state,
pharm.pharmacy_type,
rx.ndc,
mn.drug_name,
rx.generic_indicator,
md.multi_source_code,
mn.dosage_form,
rx.metric_dec_quantity,
rx.date_rx_written,
rx.max_refills,
rx.new_refill,
rx.dea,
rx.total_amount_paid
from master.rh_rx_transaction_detail rx
left join master.provider_network_v2 presc on rx.prescriber_id = presc.provider_npi
left join master.provider_network_v2_specialty ps on presc.provider_unique_id = ps.provider_unique_id
left join master.provider_network_v2 pharm on rx.pharmacy_npi = pharm.provider_npi
join master.medispan_ndc md on rx.ndc = md.ndc_upc_hri
join master.medispan_name mn on md.drug_descriptor_id = mn.drug_descriptor_id
where presc.is_current and ps.is_current and pharm.is_current
and transaction_status = 'P'
and rx.customer_id in ('100', '110', '120', '121', '130', '140', '150', '200', '210')
and substring(rx.date_filled::text from 1 for 4) = '2023';
```

The dataset contains all relevant information needed to properly identify potential fraud. Here are the created dataframe details:

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1653697 entries, 0 to 1653696
Data columns (total 32 columns):
 #   Column                         Non-Null Count    Dtype  
---  ------                         --------------    -----  
 0   masked_claim_reference_number  1653697 non-null  object 
 1   date_filled                    1653697 non-null  int64  
 2   claim_type                     1653697 non-null  object 
 3   customer_id                    1653697 non-null  int64  
 4   patient_paid_amount            0 non-null        float64
 5   copay_amount                   1653697 non-null  float64
 6   basis_of_reimbursement         1653697 non-null  object 
 7   refund                         1607527 non-null  object 
 8   transaction_status             1653697 non-null  object 
 9   masked_cardholder_id           1653697 non-null  object 
 10  age                            1653696 non-null  float64
 11  prescriber_id                  1653697 non-null  int64  
 12  prescriber_name                1653697 non-null  object 
 13  specialty_name                 1653697 non-null  object 
 14  prescriber_city                1630595 non-null  object 
 15  prescriber_state               1629430 non-null  object 
 16  pharmacy_npi                   1653697 non-null  int64  
 17  pharmacy_name                  1653697 non-null  object 
 18  physical_city                  1653697 non-null  object 
 19  physical_state                 1653099 non-null  object 
 20  pharmacy_type                  0 non-null        float64
 21  ndc                            1653697 non-null  int64  
 22  drug_name                      1653697 non-null  object 
 23  generic_indicator              1653697 non-null  object 
 24  multi_source_code              1653697 non-null  object 
 25  dosage_form                    1653697 non-null  object 
 26  metric_dec_quantity            1653697 non-null  float64
 27  date_rx_written                1653697 non-null  int64  
 28  max_refills                    1653697 non-null  int64  
 29  new_refill                     1653697 non-null  int64  
 30  dea                            1653697 non-null  int64  
 31  total_amount_paid              1653697 non-null  float64
dtypes: float64(6), int64(9), object(17)
memory usage: 403.7+ MB
```

A new column was added called is_fraud. I obtained a second dataset in the same format of claims identified as fraudulent for Q1 2023. The claims were merged and a labeled subset was created.

```
merged_df = claims_2023_df.merge(
    q1_fraud_df[['masked_claim_reference_number']],
    on='masked_claim_reference_number',
    how='left',
    indicator=True
)

merged_df['is_fraud'] = np.where(merged_df['_merge'] == 'both', 1, 0)
merged_df.drop(columns=['_merge'], inplace=True)
```

# INITIAL RESULTS
The actual q1 fraud rate was 0.38% while the predicted rate for q2-q4 was 0.11%. I did a feature analysis and noticed cardholder_id was the most important feature. I'm curious to run it again without the cardholder_id to see if the results change. 

# 24.1

# CONTINUED TRAINING
I created a second notebook to test my hypothesis that the model could be relying too heavily on cardholder_id. I kept all of the same steps as before however I removed the cardholder_id and age columns in the data preparation step. After reviewing the results I was able to determine that the fraud rate actually got worse. Feature importance appears to be the same without the two columns I removed so it did appear that those two columns did not contribute to bias. 

I was hoping to find a solution that would be close to the actual fraud rate in the control quarter (Q1). I reviewed the initial solution I came up with and decided that rather than have a static probability threshold, I would use precision recall to determine where the cutoff should be. That way, I am removing my own bias from the model and allowing the data to determine the best path forward. I also included a calibration of the model using the isotonic method. By adding the two additional steps I was able to get the rate a little bit closer at 0.26% 

# ANOTHER GOTCHA
I was happy with the results and wanted to add additional visualizations so I charted the fraud totals by customer_id. What I found was the model only predicted fraud in the same two customers that the Q1 control data had. To work around that I attempted to rebalance the data. The process returned a similar fraud rate but still only predicted fraud in the same two customers. I also noticed the rebalanced data had different important features. 

# CONCLUSION
The solutions used do a decent job predicting fraud given a control group but does appear to rely too heavily on some of the elements of the previously identified fraud. In order to ignore those patterns I would recommend we either find a more diverse group of fraud claims or eliminate them from and rely on feature engineering so the model would rely on pattern or behavior matching. 
