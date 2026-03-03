-- public.vw_telco fonte

CREATE OR REPLACE VIEW public.vw_telco
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
    NULLIF(TRIM(BOTH FROM totalcharges), ''::text)::double precision AS totalcharges,
    churn
FROM telco_raw;