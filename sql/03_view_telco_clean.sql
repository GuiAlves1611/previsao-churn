-- public.vw_telco_clean fonte

CREATE OR REPLACE VIEW public.vw_telco_clean
AS SELECT customerid,
    gender,
    seniorcitizen,
    partner,
    dependents,
    tenure,
    phoneservice,
    multiplelines,
    internetservice,
    onlinesecurity,
    onlinebackup,
    deviceprotection,
    techsupport,
    streamingtv,
    streamingmovies,
    contract,
    paperlessbilling,
    paymentmethod,
    monthlycharges,
    totalcharges,
    churn
   FROM vw_telco
  WHERE totalcharges IS NOT NULL;