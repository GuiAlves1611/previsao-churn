-- public.vw_telco_model fonte

CREATE OR REPLACE VIEW public.vw_telco_model
AS SELECT customerid,
    tenure,
    monthlycharges,
    totalcharges,
    total_services,
    avg_charge_per_service,
    seniorcitizen,
    automatic_payment,
    streaming_user,
    internet_and_phone,
    customer_lifecycle_stage,
    gender,
    partner,
    dependents,
    phoneservice,
    multiplelines,
    internetservice,
    contract,
    paperlessbilling,
    paymentmethod,
    onlinesecurity,
    onlinebackup,
    deviceprotection,
    techsupport,
    streamingtv,
    streamingmovies,
        CASE
            WHEN churn = 'Yes'::text THEN 1
            WHEN churn = 'No'::text THEN 0
            ELSE NULL::integer
        END AS target
   FROM vw_telco_features
  WHERE churn = ANY (ARRAY['Yes'::text, 'No'::text]);